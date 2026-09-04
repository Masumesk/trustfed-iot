import argparse
import pickle

from concurrent.futures import (
    ThreadPoolExecutor,
)

from client_app.api_client import (
    register_client,
    set_server_url,
)

from config import (
    NUM_CLIENTS,
    SERVER_URL,
    DATASET
)

from data.load_dataset import load_dataset

from federated.client import Client


parser = argparse.ArgumentParser()

parser.add_argument(
    "--server",
    type=str,
    default=SERVER_URL,
)

args = parser.parse_args()


set_server_url(
    args.server
)


# Load once for all clients

train_dataset, _, _ = load_dataset(DATASET ,load_test=False)


with open(
    "data/partition_cache.pkl",
    "rb",
) as f:

    client_indices = pickle.load(f)


def register_one(client_id):

    print(
        f"Registering client {client_id}"
    )

    client = Client(
        client_id=client_id,
        dataset=train_dataset,
        indices=client_indices[
            client_id
        ],
        num_classes=10,
    )

    info = (
        client.get_client_distribution()
    )

    response = register_client(
        info
    )

    print(response)

    return client_id


with ThreadPoolExecutor(
    max_workers=min(
        10,
        NUM_CLIENTS,
    )
) as executor:

    results = list(
        executor.map(
            register_one,
            range(NUM_CLIENTS),
        )
    )


print(
    "All clients registered:",
    results,
)