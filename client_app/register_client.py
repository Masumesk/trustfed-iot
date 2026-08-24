from federated.client import Client
from data import load_mnist
from data.partition import partition_dirichlet

from client_app.api_client import register_client


import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--id",
    type=int,
    required=True
)

args = parser.parse_args()

client_id = args.id


train_dataset, _ = load_mnist()


client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=30,
    alpha=0.3,
    min_samples=100,
    seed=42
)


client = Client(
    client_id=client_id,
    dataset=train_dataset,
    indices=client_indices[client_id],
    num_classes=10
)


info = client.get_client_distribution()


response = register_client(info)


print(response)