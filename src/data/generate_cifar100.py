import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from torchvision import datasets, transforms
import os


def create_cifar100_subset(dataset, num_examples_by_class=10):
    """
    Crée un sous-dataset CIFAR100 avec un nombre spécifique d'exemples par classe
    compatible avec DataLoader classique
    """
    examples_by_class = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Collecter les exemples par classe
    list_indexes = []
    for i, (images, labels) in enumerate(dataloader):
        label = labels.item()

        if len(examples_by_class[label]) < num_examples_by_class:
            examples_by_class[label].append((images.squeeze(0), label))
            list_indexes.append(i)

        # Vérifier si on a assez d'exemples pour toutes les classes (100 pour CIFAR100)
        if all(len(examples_by_class[i]) == num_examples_by_class for i in range(100)):
            break

    print(f"Indexes sélectionnés pour le sous-dataset : {list_indexes}")

    # Convertir en listes plates
    all_images = []
    all_labels = []

    for class_label in sorted(examples_by_class.keys()):
        for image, label in examples_by_class[class_label]:
            all_images.append(image)
            all_labels.append(label)

    # Créer des tenseurs
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)

    # Créer un TensorDataset compatible avec DataLoader
    subset_dataset = TensorDataset(images_tensor, labels_tensor)

    print(f"Sous-dataset CIFAR100 créé avec {len(subset_dataset)} exemples")
    print(f"Sauvegardé dans 'cifar100_subset_{num_examples_by_class}_per_class.pth'")

    return subset_dataset


# Fonction pour charger le sous-dataset
def load_cifar100_subset(num_examples_by_class=10):
    """
    Charge un sous-dataset CIFAR100 précédemment créé
    """
    data = torch.load(
        f"data/datasets/cifar100_subset_{num_examples_by_class}_per_class.pth"
    )
    return data["dataset"]


# Fonction générique pour n'importe quel dataset
def create_dataset_subset(
    dataset, num_classes, num_examples_by_class=10, dataset_name="custom"
):
    """
    Version générique pour créer un sous-dataset avec n'importe quel nombre de classes
    """
    examples_by_class = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    list_indexes = []
    # Collecter les exemples par classe
    for i, (images, labels) in enumerate(dataloader):
        label = labels.item()

        if len(examples_by_class[label]) < num_examples_by_class:
            examples_by_class[label].append((images.squeeze(0), label))
            list_indexes.append(i)

        # Vérifier si on a assez d'exemples pour toutes les classes
        if all(
            len(examples_by_class[i]) == num_examples_by_class
            for i in range(num_classes)
        ):
            break

    # Convertir en listes plates
    all_images = []
    all_labels = []

    for class_label in sorted(examples_by_class.keys()):
        for image, label in examples_by_class[class_label]:
            all_images.append(image)
            all_labels.append(label)

    # Créer des tenseurs
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)

    # Créer un TensorDataset compatible avec DataLoader
    subset_dataset = TensorDataset(images_tensor, labels_tensor)

    print(f"Sous-dataset {dataset_name} créé avec {len(subset_dataset)} exemples")
    print(
        f"Sauvegardé dans '{dataset_name}_subset_{num_examples_by_class}_per_class.pth'"
    )

    return subset_dataset


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    print("Téléchargement de CIFAR100...")
    cifar100_dataset_TRAIN = datasets.CIFAR100(
        "data", train=True, download=True, transform=transform
    )
    cifar100_dataset_TEST = datasets.CIFAR100(
        "data", train=False, download=True, transform=transform
    )

    print("CIFAR100 téléchargé avec succès !")

    print("\nCréation du sous-dataset CIFAR100...")
    subset = create_cifar100_subset(cifar100_dataset_TRAIN, num_examples_by_class=10)

    subset_loader = DataLoader(subset, batch_size=32, shuffle=True)

    print(f"\nTest du DataLoader:")
    for batch_idx, (images, labels) in enumerate(subset_loader):
        print(f"Batch {batch_idx + 1}: {images.shape}, labels: {labels.shape}")
        if batch_idx == 2:
            break

    print(f"\nTotal d'exemples dans le sous-dataset: {len(subset)}")
    print(f"Nombre de classes uniques: {len(torch.unique(subset.tensors[1]))}")
