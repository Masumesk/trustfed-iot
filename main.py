from clustering.hellinger import hellinger_distance, build_distance_matrix
from clustering.optics import optic_clustering,create_clusters,calculate_medoids,assign_noise

from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client


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
    # print(
    #     f"Client {client_id}: "
    #     f"{len(indices)} samples"
    # )

# client = clients[0]

# print(client)
# print(client.num_samples)
# print(client.histogram)
# print(client.distribution)

# distance = hellinger_distance(
#     clients[18].distribution,
#     clients[2].distribution
# )
#
# print(
#     "Hellinger distance between "
#     "Client 0 and Client 1:",
#     distance
# )

distance_matrix = build_distance_matrix(clients)

# print(distance_matrix)
# print("Shape:", distance_matrix.shape)

for client in clients:
    print(client.histogram)

labels = optic_clustering(
    distance_matrix,
    min_samples=2,
    xi=0.02,
    min_cluster_size=None
)

print("labels:")
print(labels)

G, Q = create_clusters(labels)

print("clusters:")
print(G)

print("Noise:")
print(Q)

medoids = calculate_medoids(G,distance_matrix)

print("medoids:")
print(medoids)

G_final = assign_noise(G,Q,medoids,distance_matrix,assignment_threshold=0.6)

print("all clusters:")
print(G_final)

