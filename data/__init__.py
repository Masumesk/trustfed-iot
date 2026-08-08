from torchvision import datasets, transforms


def load_mnist(root="./datasets"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=False,
        transform=transform
    )

    return train_dataset, test_dataset