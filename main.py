import numpy as np

from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client
from federated.server import Server
from federated.model_update import compute_model_update
from models.model import MNISTCNN

train_dataset, test_dataset = load_mnist()

client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=30,
    alpha=0.3,
    min_samples=100,
    seed=42
)

clients = []

for client_id, indices in enumerate(client_indices):

    client = Client(
        client_id=client_id,
        dataset=train_dataset,
        indices=indices,
        num_classes=10
    )

    clients.append(client)

clients_by_id = {
    client.client_id: client
    for client in clients
}


server = Server()

client_infos = {}

for client in clients:
    info = client.get_client_distribution()
    client_infos[client.client_id] = info

server.receive_client_distributions(client_infos)

server.client_clustering()

print("Clusters:")
print(server.clusters)

print("Cluster sample counts:")
print(server.cluster_samples_counts)

print("Cluster data shares:")
print(server.cluster_data_shares)

server.main_and_backup_client_selection()
print(server.main_clients)
print(server.backup_clients)

global_model = MNISTCNN()


for cluster_id, client_ids in server.main_clients.items():

    for client_id in client_ids:

        client = clients_by_id[client_id]

        local_model, loss = client.local_train(
            model=global_model,
            epochs=1,
            batch_size=32,
            lr=0.01
        )

        update = compute_model_update(
            global_model,
            local_model
        )

        server.receive_client_update(
            client_id,
            update
        )

        print(
            f"MAIN | "
            f"Cluster {cluster_id} | "
            f"Client {client_id} | "
            f"Loss = {loss:.4f} | "
            f"Update size = {update.shape}"
        )


for cluster_id, client_ids in server.backup_clients.items():

    for client_id in client_ids:
        client = clients_by_id[client_id]

        local_model, loss = client.local_train(
            model=global_model,
            epochs=1,
            batch_size=32,
            lr=0.01
        )

        update = compute_model_update(
            global_model,
            local_model
        )

        server.receive_client_update(
            client_id,
            update
        )

        print(
            f"BACKUP | "
            f"Cluster {cluster_id} | "
            f"Client {client_id} | "
            f"Loss = {loss:.4f} | "
            f"Update size = {update.shape}"
        )

print("Before:")
print(server.trust_scores)

server.evaluate_client_trust()

print("After:")
print(server.trust_scores)
