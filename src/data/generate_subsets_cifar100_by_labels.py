import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
from collections import defaultdict


def create_cifar100_incremental_datasets(
    dataset, train: bool = True, save_dir="data/datasets/cifar100_subsets"
):
    """
    Crée les datasets CIFAR100 incrémentaux:
    - cifar100-0-9: classes 0-9
    - cifar100-0-19: classes 0-19
    - ...
    - cifar100-0-99: classes 0-99 (dataset complet)
    """

    os.makedirs(save_dir, exist_ok=True)

    print("Collecte de tous les exemples par classe...")
    examples_by_class = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, labels in dataloader:
        label = labels.item()
        examples_by_class[label].append((images.squeeze(0), label))

    print(f"Collecte terminée. Classes disponibles: {sorted(examples_by_class.keys())}")

    for max_class in range(9, 100, 10):
        print(f"\nCréation des datasets pour classes 0-{max_class}...")

        full_images = []
        full_labels = []

        for class_label in range(max_class + 1):
            if class_label in examples_by_class:
                class_examples = examples_by_class[class_label]

                for image, label in class_examples:
                    full_images.append(image)
                    full_labels.append(label)

        full_images_tensor = torch.stack(full_images)
        full_labels_tensor = torch.tensor(full_labels)

        full_dataset = TensorDataset(full_images_tensor, full_labels_tensor)

        full_name = f"cifar100-0-{max_class}_" + ("train" if train else "test")

        torch.save(
            {
                "images": full_images_tensor,
                "labels": full_labels_tensor,
                "dataset": full_dataset,
                "num_classes": max_class + 1,
                "num_examples": len(full_dataset),
            },
            os.path.join(save_dir, f"{full_name}.pth"),
        )

        print(f"✓ {full_name}: {len(full_dataset)} exemples, {max_class + 1} classes")

    if train:
        for max_class in range(9, 100, 10):
            print(f"\nCréation des datasets pour classes 0-{max_class}...")

            subset_images = []
            subset_labels = []

            for class_label in range(max_class + 1):
                if class_label in examples_by_class:
                    class_examples = examples_by_class[class_label]

                    for image, label in class_examples[:10]:
                        subset_images.append(image)
                        subset_labels.append(label)

            subset_images_tensor = torch.stack(subset_images)
            subset_labels_tensor = torch.tensor(subset_labels)

            subset_dataset = TensorDataset(subset_images_tensor, subset_labels_tensor)

            subset_name = f"cifar100-0-{max_class}-subset"

            torch.save(
                {
                    "images": subset_images_tensor,
                    "labels": subset_labels_tensor,
                    "dataset": subset_dataset,
                    "num_classes": max_class + 1,
                    "num_examples": len(subset_dataset),
                },
                os.path.join(save_dir, f"{subset_name}.pth"),
            )

            print(
                f"✓ {subset_name}: {len(subset_dataset)} exemples, {max_class + 1} classes"
            )


def load_cifar100_incremental_dataset(
    max_class, subset=False, save_dir="data/datasets/cifar100_subsets"
):
    if subset:
        filename = f"cifar100-0-{max_class}-subset.pth"
    else:
        filename = f"cifar100-0-{max_class}.pth"

    filepath = os.path.join(save_dir, filename)
    data = torch.load(filepath)

    print(
        f"Chargé {filename}: {data['num_examples']} exemples, {data['num_classes']} classes"
    )
    return data["dataset"]


def list_available_datasets(save_dir="data/datasets/cifar100_subsets"):
    import glob

    pattern = os.path.join(save_dir, "cifar100-0-*.pth")
    files = glob.glob(pattern)

    print("Datasets CIFAR100 incrémentaux disponibles:")
    for file in sorted(files):
        basename = os.path.basename(file)
        data = torch.load(file)
        print(
            f"  {basename}: {data['num_examples']} exemples, {data['num_classes']} classes"
        )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/datasets/cifar100_subsets", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    print("Téléchargement de CIFAR100...")
    cifar100_dataset_train = datasets.CIFAR100(
        "data", train=True, download=True, transform=transform
    )
    cifar100_dataset_test = datasets.CIFAR100(
        "data", train=False, download=True, transform=transform
    )
    print("CIFAR100 téléchargé avec succès !")

    print("\nCréation des datasets incrémentaux...")
    create_cifar100_incremental_datasets(cifar100_dataset_train, train=True)
    create_cifar100_incremental_datasets(cifar100_dataset_test, train=False)

    print("\n" + "=" * 50)
    list_available_datasets()
