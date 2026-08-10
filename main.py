# from clustering.hellinger import hellinger_distance, build_distance_matrix
# from clustering.optics import optic_clustering,create_clusters,calculate_medoids,assign_noise
# from clustering.cluster_analysis import calculate_cluster_samples_share,calculate_cluster_samples_counts_size


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

G = server.client_clustering()

print("\nClusters:")
print(G)

print("\nCluster sample counts:")
print(server.cluster_samples_counts)

print("\nCluster data shares:")
print(server.cluster_data_shares)

