import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, random_split, DataLoader
from collections import defaultdict

# torch.serialization.add_safe_globals([TensorDataset])


def generate_mnist(n_samples: int = 10000, test_split: float = 0.2, save: bool = False):
    """
    Version corrigée avec séparation train/test

    Args:
        n_samples: Nombre total d'échantillons
        test_split: Proportion pour le test (0.2 = 20%)
        save: Sauvegarder ou non

    Returns:
        train_dataset, test_dataset (si save=False)
    """

    print(f"Génération de {n_samples} échantillons MNIST...")
    print(
        f"Séparation : {int(n_samples*(1-test_split))} train, {int(n_samples*test_split)} test"
    )

    # 1. Charger le dataset MNIST original
    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(
        root="./data/datasets/", train=True, download=True, transform=transform
    )

    # 2. Extraire les données
    images = mnist_train.data[:n_samples].float().unsqueeze(1) / 255.0  # [0,1] range
    labels = mnist_train.targets[:n_samples].long()

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print("Values range ; ", images.min().item(), images.max().item())

    # 3. Créer le dataset complet
    full_dataset = TensorDataset(images, labels)

    # 4. SÉPARATION TRAIN/TEST
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_test

    train_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),  # Reproductibilité
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 5. Sauvegarder si demandé
    if save:
        # Sauvegarder des exemples de chaque classe
        create_mnist_subset_save_indices(full_dataset, 10)

    return train_dataset, test_dataset


def create_mnist_subset(dataset, num_examples_by_class=10):
    """
    Crée un sous-dataset MNIST avec un nombre spécifique d'exemples par classe
    compatible avec DataLoader classique
    """
    examples_by_class = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Collecter les exemples par classe
    for images, labels in dataloader:
        label = labels.item()

        if len(examples_by_class[label]) < num_examples_by_class:
            examples_by_class[label].append((images.squeeze(0), label))

        # Vérifier si on a assez d'exemples pour toutes les classes
        if all(len(examples_by_class[i]) == num_examples_by_class for i in range(10)):
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

    # Sauvegarder
    torch.save(
        {"dataset": subset_dataset},
        f"data/datasets/mnist_subset_{num_examples_by_class}_per_class.pth",
    )

    print(f"Sous-dataset créé avec {len(subset_dataset)} exemples")
    print(f"Sauvegardé dans 'mnist_subset_{num_examples_by_class}_per_class.pth'")

    return subset_dataset


def create_mnist_subset_save_indices(dataset, num_examples_by_class=10):
    """
    Crée un sous-dataset MNIST avec un nombre spécifique d'exemples par classe
    compatible avec DataLoader classique
    Retourne aussi les index des exemples sélectionnés dans le dataset original
    """
    examples_by_class = defaultdict(list)
    indices_by_class = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Collecter les exemples par classe avec leurs index
    for idx, (images, labels) in enumerate(dataloader):
        label = labels.item()

        if len(examples_by_class[label]) < num_examples_by_class:
            examples_by_class[label].append((images.squeeze(0), label))
            indices_by_class[label].append(idx)

        # Vérifier si on a assez d'exemples pour toutes les classes
        if all(len(examples_by_class[i]) == num_examples_by_class for i in range(10)):
            break

    # Convertir en listes plates
    all_images = []
    all_labels = []
    all_indices = []

    for class_label in sorted(examples_by_class.keys()):
        for image, label in examples_by_class[class_label]:
            all_images.append(image)
            all_labels.append(label)
        all_indices.extend(indices_by_class[class_label])

    # Créer des tenseurs
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)
    indices_tensor = torch.tensor(all_indices)

    # Créer un TensorDataset compatible avec DataLoader
    subset_dataset = TensorDataset(images_tensor, labels_tensor)

    print(f"Sous-dataset créé avec {len(subset_dataset)} exemples")
    print(f"Index originaux : {all_indices}")
    print(f"Sauvegardé dans 'mnist_subset_{num_examples_by_class}_per_class.pth'")

    return subset_dataset, indices_tensor


if __name__ == "__main__":
    generate_mnist(save=True)
