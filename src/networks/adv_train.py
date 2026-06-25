import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from pydantic import ValidationError
import gc

from adversarial_attacks import (
    PGDAttack,
    SDPAttack,
    LPAttack3Parallel,
    CrownIBP_Attack,
)
from fastsdp_tools import Adversarial_Network_Training, get_project_path


def evaluate_clean(model, testloader, device):
    """Évaluation sur données propres"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def evaluate_robust(model, testloader, device, pgd_config, max_batches=None):
   
    model.eval()

    eval_pgd_config = pgd_config.copy()
    eval_pgd_config["steps"] = 20  

    pgd_attack = PGDAttack(
        model=model,
        eps=eval_pgd_config["eps"],
        alpha=eval_pgd_config["alpha"],
        steps=eval_pgd_config["steps"],
        random_start=eval_pgd_config["random_start"],
        norm=eval_pgd_config["norm"],
    )

    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(testloader):

        if max_batches and batch_idx >= max_batches:
            break
        if batch_idx >= 100:
            break
        print("Evaluating batch ", batch_idx)
        inputs, labels = inputs.to(device), labels.to(device)


        with torch.enable_grad():
            adv_inputs = pgd_attack.forward(inputs, labels)

        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



def complex_adversarial_training_loop(
    model,
    trainloader,
    testloader,
    robust_datas_loader,
    device,
    adversarial_attack: str = "PGD",
    lambda_: float = 1,
    project_name: str = "adversarial-training",
    experiment_name: str = None,
    log_frequency: int = 1,  # Log tous les N epochs
    use_wandb: bool = False,
    epsilon_to_test: float = None,
    **kwargs,
):
    """
    Entraînement adversarial avec logging W&B

    Args:
        use_wandb: Activer/désactiver W&B
        project_name: Nom du projet W&B
        experiment_name: Nom de l'expérience
        log_frequency: Fréquence de logging (epochs)
    """
    yaml_file = kwargs.get("yaml_file", "adversarial_network_training.yaml")
    print("KWARGS  : ", kwargs)
    lr = kwargs.get("lr", 0.001)
    eps = kwargs.get("epsilon")
    num_epochs = kwargs.get("num_epochs", 100)
    print("lambda : ", lambda_)
    if use_wandb:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                "learning_rate": lr,
                "epsilon": eps,
                "num_epochs": num_epochs,
                "adversarial_attack": adversarial_attack,
                "lambda": lambda_,
                "model_architecture": str(model),
                "batch_size": trainloader.batch_size,
            },
            save_code=False,
            dir="wandb",
            mode="offline",
            settings=wandb.Settings(_disable_stats=True),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    print("Adversarial training with attack: ", adversarial_attack)

    if adversarial_attack == "PGD":
        alpha = kwargs.get("alpha")
        steps = kwargs.get("steps")
        random_start = kwargs.get("random_start")
        attack = PGDAttack(
            model=model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            norm="inf",
        )
        print("PGD Attack configured with eps:", eps, "alpha:", alpha, "steps:", steps)
        print(attack)

    elif adversarial_attack == "LP":
        attack = LPAttack3Parallel(
            model=model,
            num_classes=model.n[-1],
            eps=eps,
            targeted=False,
            norm="inf",
            compute_bounds_method=kwargs.get("compute_bounds_method", "alpha-CROWN"),
        )
    elif adversarial_attack == "SDP":
        attack = SDPAttack(
            model=model,
            num_classes=model.n[-1],
            eps=eps,
            norm="inf",
        )

    elif adversarial_attack == "CROWN-IBP":
        attack = CrownIBP_Attack(
            model=model,
            shape=kwargs.get("shape", (1, 1, 28, 28)),  # Exemple pour MNIST
            device=device,
            epsilon=eps,
            kappa=1.0,
            criterion=criterion,
        )
 
    train_losses = []
    clean_accuracies = []
    robust_accuracies_EPS_0_3 = []
    robust_accuracies_EPS_0_1 = []
    robust_accuracies_EPS_0_0_1 = []
    robust_accuracies_EPS_to_test = []
    epochs_logged = []

    model.to(device)

    for epoch in range(num_epochs):
        print(f"Training epoch: {epoch}")
        model.train()
        running_loss = 0.0
        batch_losses = []

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            print(f"    Processing batch {batch_idx + 1}/{len(trainloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            model.eval() 

            if adversarial_attack == "PGD":
                with torch.enable_grad():
                    adv_inputs = attack.forward(inputs, labels)
                model.train()  
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)

            elif adversarial_attack == "LP" or adversarial_attack == "SDP":

                if adversarial_attack == "LP":
                    W, b = model.extract_weights()
                    loss_pertubation, adv_inputs = attack(
                        inputs, labels, model.K, model.n, W, b
                    )
                    print("Adversarial inputs shape: ", adv_inputs.shape)

                loss_pertubation = torch.tensor(
                    loss_pertubation,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=device,
                )

                model.train()
            
                print("Inputs shape: ", inputs.shape)
                print("Labels shape: ", labels.shape)
                outputs = model(inputs)
                print("Outputs shape: ", outputs.shape)
                loss = criterion(outputs, labels) + lambda_ * loss_pertubation

            elif adversarial_attack == "CROWN-IBP":
                loss = attack.compute_crown_ibp_loss(inputs, labels)
                print("CROWN-IBP loss:", loss.item())

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)

            pbar.set_postfix({"Loss": f"{batch_loss:.4f}"})


        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        if epoch % log_frequency == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_train_loss:.4f}")

            model.eval()
            clean_acc = evaluate_clean(model, testloader, device)
            clean_accuracies.append(clean_acc)
            print(f"Clean Accuracy: {clean_acc:.2%}")

            robust_acc = None
            if epoch % (log_frequency * 2) == 0: 
                robust_acc = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": eps,
                        "alpha": eps / 4,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_3.append(robust_acc)
                print(f"Robust Accuracy EPS=0.3: {robust_acc:.2%}")

                robust_acc_eps_0_1 = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": 0.1,
                        "alpha": 0.01,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_1.append(robust_acc_eps_0_1)

                robust_acc_eps_0_0_1 = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": 0.01,
                        "alpha": 0.001,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_0_1.append(robust_acc_eps_0_0_1)

                robust_acc_eps_to_test = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": epsilon_to_test,
                        "alpha": 0.001,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_to_test.append(robust_acc_eps_to_test)

            epochs_logged.append(epoch)

            if use_wandb:
                log_dict = {
                    "train_loss": avg_train_loss,
                    "clean_accuracy": clean_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                if robust_acc is not None:
                    log_dict[f"robust_accuracy_epsilon={eps}"] = robust_acc
                    log_dict["robust_accuracy_epsilon=0.1"] = robust_acc_eps_0_1
                    log_dict["robust_accuracy_epsilon=0.01"] = robust_acc_eps_0_0_1
                    log_dict[f"robust_accuracy_epsilon={epsilon_to_test}"] = (
                        robust_acc_eps_to_test
                    )

                wandb.log(log_dict)

        torch.cuda.empty_cache()
        gc.collect()

    if not use_wandb:
        plot_training_curves(
            epochs_logged, train_losses, clean_accuracies, robust_accuracies_EPS_0_3
        )


    if use_wandb:
        wandb.finish()

    return {
        "train_losses": train_losses,
        "clean_accuracies": clean_accuracies,
        "robust_accuracies": robust_accuracies_EPS_0_3,
        "epochs": epochs_logged,
    }


def plot_training_curves(epochs, train_losses, clean_accs, robust_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, clean_accs, "g-", label="Clean Accuracy")
    if robust_accs:
        robust_epochs = epochs[::2]  # Moins fréquent
        ax2.plot(robust_epochs, robust_accs, "r-", label="Robust Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_adversarial_training_config(config_path):
    """
    Load adversarial trianing config from yaml file.
    """

    with open(get_project_path(config_path), "r") as file:
        raw_config = yaml.safe_load(file)

    try:
        validated_config = Adversarial_Network_Training(**raw_config)
    except ValidationError as e:
        raise

    return validated_config.dict()
