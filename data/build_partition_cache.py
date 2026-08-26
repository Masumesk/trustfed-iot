import pickle
from data import load_mnist
from data.partition import partition_dirichlet

train_dataset, _ = load_mnist()

client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=30,
    alpha=0.3,
    min_samples=100,
    seed=42
)

with open("data/partition_cache.pkl", "wb") as f:
    pickle.dump(client_indices, f)

print("Partition cached.")