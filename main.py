import numpy as np

from data.__init__ import load_mnist
from data.partition import (
    partition_dirichlet,
    create_client_subsets
)

train_dataset, test_dataset = load_mnist()

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=20,
    alpha=0.3,
    min_samples=100,
    seed=42
)

clients = create_client_subsets(
    train_dataset,
    client_indices
)

print("Number of clients:", len(clients))

for client_id, indices in enumerate(client_indices):

    labels = np.array(
        train_dataset.targets
    )[indices]

    histogram = np.bincount(
        labels,
        minlength=10
    )

    print(
        f"Client {client_id}: "
        f"{len(indices)} samples"
    )

    print(
        "Histogram:",
        histogram
    )

    print()