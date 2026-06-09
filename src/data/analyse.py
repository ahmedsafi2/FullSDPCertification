from collections import Counter
from torch.utils.data import Subset


def analyze_class_distribution(dataset, dataset_name="Dataset"):
    """Analyze and visualize class distribution in dataset"""

    # Get all labels from the dataset
    labels = []

    if isinstance(dataset, Subset):
        # If it's a subset, get labels from the original dataset
        if hasattr(dataset.dataset, "targets"):
            labels = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            # Fallback: iterate through the subset
            for i in dataset.indices:
                _, label = dataset.dataset[i]
                labels.append(label)
    else:
        # For other dataset types
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
            # For TensorDataset
            labels = dataset.tensors[1].tolist()
        else:
            # Fallback: iterate through the entire dataset (slower but works)
            print(f"Extracting labels from {dataset_name}... (this may take a moment)")
            for i in range(len(dataset)):
                _, label = dataset[i]
                labels.append(label)

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get CIFAR-100 class names
    cifar100_classes = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    # Print statistics
    print(f"\n{dataset_name} Class Distribution:")
    print(f"Total samples: {len(labels)}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Classes present: {sorted(class_counts.keys())}")
    print("\nClass breakdown:")

    for class_id in sorted(class_counts.keys()):
        class_name = (
            cifar100_classes[class_id]
            if class_id < len(cifar100_classes)
            else f"Class_{class_id}"
        )
        count = class_counts[class_id]
        percentage = (count / len(labels)) * 100
        print(
            f"  Class {class_id:2d} ({class_name:15s}): {count:4d} samples ({percentage:5.1f}%)"
        )

    return class_counts
