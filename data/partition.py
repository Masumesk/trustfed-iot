import numpy as np
from torch.utils.data import Subset

from config import NUM_CLIENTS, DIRICHLET_ALPHA, MIN_SAMPLES, DATA_SEED


def partition_dirichlet(
    dataset,
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA,
    min_samples=MIN_SAMPLES,
    seed=DATA_SEED
):

   # Split dataset among clients using Dirichlet Label Skew.

    rng = np.random.default_rng(seed)


    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[
            dataset.indices
        ]
    else:
        targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    for _ in range(100):

        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):

            class_indices = np.where(targets == class_id)[0]

            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.repeat(alpha, num_clients)
            )

            split_points = (
                np.cumsum(proportions)[:-1]
                * len(class_indices)
            ).astype(int)

            splits = np.split(
                class_indices,
                split_points
            )

            for client_id, indices in enumerate(splits):
                client_indices[client_id].extend(
                    indices.tolist()
                )

        for client_id in range(num_clients):
            rng.shuffle(client_indices[client_id])

        client_sizes = [
            len(indices)
            for indices in client_indices
        ]

        if min(client_sizes) >= min_samples:
            return client_indices

    raise RuntimeError(
        "Could not create Dirichlet partition "
        "with the requested minimum client size."
    )

def partition_iid(
        dataset,
        num_clients=NUM_CLIENTS,
        seed=DATA_SEED
):
    rng=np.random.default_rng(seed)

    indices=np.arange(len(dataset))
    rng.shuffle(indices)

    client_indices=np.array_split(
        indices,num_clients
    )

    return [
        client_indices[i].tolist()
        for i in range(num_clients)
    ]
