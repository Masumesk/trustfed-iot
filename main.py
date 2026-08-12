# from clustering.hellinger import hellinger_distance, build_distance_matrix
# from clustering.optics import optic_clustering,create_clusters,calculate_medoids,assign_noise
# from clustering.cluster_analysis import calculate_cluster_samples_share,calculate_cluster_samples_counts_size
import numpy as np

from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client
from federated.server import Server


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


#fake updates
for cluster_id, clients in server.main_clients.items():

    for i, client_id in enumerate(clients):

        if client_id == 28:
                    update = np.array([
                        1000,
                        -100.0,
                        20.0
                    ])
        else:
            update = np.array([
                0.1 + i * 0.01,
                0.2 + i * 0.01,
                0.3 + i * 0.01
            ])

        server.receive_client_update(
            client_id,
            update
        )


for cluster_id, clients in server.backup_clients.items():

    for i, client_id in enumerate(clients):

       
        update = np.array([
            0.11 + i * 0.01,
            0.21 + i * 0.01,
            0.31 + i * 0.01
        ])

        server.receive_client_update(
            client_id,
            update
        )


print("Before:")
print(server.trust_scores)

server.evaluate_client_trust()

print("After:")
print(server.trust_scores)
