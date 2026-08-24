import argparse

from federated.client import Client
from data import load_mnist
from data.partition import partition_dirichlet
from federated.model_update import compute_model_update
from client_app.api_client import (
    register_client,
    get_training_package,
    load_global_model,
    send_update
)

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
    num_clients=30,  #Define the number above
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

package = get_training_package(client_id)

model = load_global_model(package)

print("global model received")

local_model, loss = client.local_train(
    model=model,
    epochs=package["local_epochs"],
    batch_size=package["batch_size"],
    lr=package["learning_rate"]
)

update = compute_model_update(
    model,
    local_model
)

response = send_update(
    client_id,
    update
)

print(response)