import pickle
from data.load_dataset import load_dataset
from data.partition import partition_dirichlet
from config import (
    NUM_CLIENTS,
    DIRICHLET_ALPHA,
    MIN_SAMPLES,
    DATA_SEED,
    DATASET,
    
)

train_dataset, _, _ = load_dataset(DATASET)

client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA,
    min_samples=MIN_SAMPLES,
    seed=DATA_SEED
)

with open("data/partition_cache.pkl", "wb") as f:
    pickle.dump(client_indices, f)

print("Partition cached.")