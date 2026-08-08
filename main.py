from clustering.hellinger import hellinger_distance, build_distance_matrix
from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client


train_dataset, test_dataset = load_mnist()

client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=20,
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

client = clients[0]

print(client)
print(client.num_samples)
print(client.histogram)
print(client.distribution)

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

print(distance_matrix)
print("Shape:", distance_matrix.shape)