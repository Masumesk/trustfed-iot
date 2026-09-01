from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np


def load_dataset(name,root="./datasets", val_ratio=0.2, seed=42):

    if name == "MNIST":
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

    elif name == "CIFAR10":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.CIFAR10(
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