# from clustering.hellinger import hellinger_distance, build_distance_matrix
# from clustering.optics import optic_clustering,create_clusters,calculate_medoids,assign_noise
# from clustering.cluster_analysis import calculate_cluster_samples_share,calculate_cluster_samples_counts_size
from colorama.ansi import clear_line
import torch
from models.model import MNISTCNN
from data.partition import partition_iid

from data import load_mnist
# from data.partition import partition_dirichlet
from federated.client import Client
from federated.server import Server
from torch.utils.data import DataLoader
import torch.nn as nn


train_dataset, test_dataset = load_mnist()

# client_indices = partition_dirichlet(
#     dataset=train_dataset,
#     num_clients=30,
#     alpha=0.3,
#     min_samples=100,
#     seed=42
# )
client_indices=partition_iid(dataset=train_dataset)
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

# server = Server(clients)
# G = server.client_clustering(min_samples=2,xi=0.02,min_cluster_size=None,assignment_threshold=0.6)
# print(G)
# print(server.cluster_data_shares)

client = clients[0]
client_dataset=client.get_subset()
client_loader=DataLoader(
    client_dataset,
    batch_size=32,
    shuffle=True

)

# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)
# print("Labels:", labels)

model=MNISTCNN()


local_model, loss = clients[0].local_train(
    model=model,
    epochs=1,
    batch_size=32,
    lr=0.01
)

print("Client 0 loss:", loss)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)
local_model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = local_model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

accuracy = correct / total

print("Test accuracy:", accuracy)
#
# optimizer.zero_grad()

# outputs = model(images)

# loss = criterion(outputs, labels)
#
# loss.backward()

# optimizer.step()

# print(loss.item())
# for client in clients:
#     print(
#         client.client_id,
#     client.num_samples,
#
#     client.distribution,
#
#     )
#
#     print (model)

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

# distance_matrix = build_distance_matrix(clients)

# # print(distance_matrix)
# # print("Shape:", distance_matrix.shape)

# # for client in clients:
#     # print(client.histogram)

# labels = optic_clustering(
#     distance_matrix,
#     min_samples=2,
#     xi=0.02,
#     min_cluster_size=None
# )

# # print("labels:")
# # print(labels)

# G, Q = create_clusters(labels)

# # print("clusters:")
# # print(G)

# # print("Noise:")
# # print(Q)

# medoids = calculate_medoids(G,distance_matrix)

# # print("medoids:")
# # print(medoids)

# G_final = assign_noise(G,Q,medoids,distance_matrix,assignment_threshold=0.6)

# # print("all clusters:")
# # print(G_final)

# cluster_samples_counts = calculate_cluster_samples_counts_size(G_final,clients)

# # Calculate the data share of each cluster
# cluster_share = calculate_cluster_samples_share(cluster_samples_counts)

# print("cluster counts:")
# print(cluster_samples_counts)

# print("cluster data shares:")
# print(cluster_share)

