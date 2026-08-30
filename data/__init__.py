from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np


def load_mnist(root="./datasets", val_ratio=0.2, seed=42):
    """
    Load MNIST with train/val/test split.
    - Train: 80% of original train set (for federated learning)
    - Val: 20% of original train set (for parameter tuning)
    - Test: original test set (for final evaluation only)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    # Split train into train + validation
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_dataset))
    val_size = int(len(train_dataset) * val_ratio)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    return train_subset, val_subset, test_dataset