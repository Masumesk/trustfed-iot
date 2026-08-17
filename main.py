import numpy as np
import torch

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


global_model = MNISTCNN()

NUM_ROUNDS = 2
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01


for round_id in range(1, NUM_ROUNDS + 1):

    print(f"round {round_id}")

   
    server.start_round(round_id)
    server.main_and_backup_client_selection()

    print("Main clients:")
    print(server.main_clients)

    print("Backup clients:")
    print(server.backup_clients)


    training_package = server.create_training_package(
        global_model=global_model,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )


    for cluster_id, client_ids in server.main_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            client.receive_training_package(training_package)


    for cluster_id, client_ids in server.backup_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            client.receive_training_package(training_package)


    
    for cluster_id, client_ids in server.main_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            local_model, loss = (client.train_received_package())

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
                f"Round {round_id} | "
                f"Cluster {cluster_id} | "
                f"Client {client_id} | "
                f"Loss = {loss:.4f}"
            )


    for cluster_id, client_ids in server.backup_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            local_model, loss = (client.train_received_package())

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
                f"Round {round_id} | "
                f"Cluster {cluster_id} | "
                f"Client {client_id} | "
                f"Loss = {loss:.4f}"
            )


    print("Trust scores before evaluation:")
    print(server.trust_scores)

    server.trust_evaluation_and_backup_replacement()

    print("Trust scores after evaluation:")
    print(server.trust_scores)

    print("Accepted clients:")
    print(server.accepted_clients)


    global_model = server.aggregate(
        global_model,
        trim_ratio=0.2
    )

    print(f"Round {round_id} completed.")
    print("---------")
    

