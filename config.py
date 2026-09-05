import os
import numpy as np


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _env_str(name, default):
    return os.getenv(name, str(default))


# Server
SERVER_URL = _env_str("SERVER_URL", "http://127.0.0.1:8000")


# Federated configuration

NUM_CLIENTS = _env_int("NUM_CLIENTS", 40)
PARTICIPATION_RATIO = _env_float("PARTICIPATION_RATIO", 0.4)

NUM_ROUNDS = _env_int("NUM_ROUNDS", 100)

LOCAL_EPOCHS = _env_int("LOCAL_EPOCHS", 2) 
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
# LEARNING_RATE = _env_float("LEARNING_RATE", 0.01)

CLIENT_WORKERS = _env_int(
    "CLIENT_WORKERS",
    2,
)


# Reproducibility
MODEL_SEED = _env_int("MODEL_SEED", 42)
TRAINING_SEED = _env_int("TRAINING_SEED", 42)
SELECTION_SEED = _env_int("SELECTION_SEED", 42)


# Data distribution
VAL_RATIO = _env_float("VAL_RATIO", 0.2)


DIRICHLET_ALPHA = _env_float("DIRICHLET_ALPHA", 0.5)
MIN_SAMPLES = _env_int("MIN_SAMPLES", 100)
DATA_SEED = _env_int("DATA_SEED", 42)


# Attack configuration

ATTACK_TYPE = os.getenv("ATTACK_TYPE", None)
MALICIOUS_RATIO = _env_float("MALICIOUS_RATIO", 0.2)
MALICIOUS_SEED = _env_int("MALICIOUS_SEED", 42)


def get_malicious_ids():
    rng = np.random.RandomState(MALICIOUS_SEED)

    num_malicious = int(
        NUM_CLIENTS * MALICIOUS_RATIO
    )

    malicious_ids = rng.choice(
        NUM_CLIENTS,
        num_malicious,
        replace=False,
    )

    return set(
        int(client_id)
        for client_id in malicious_ids
    )



# OPTICS / clustering
OPTICS_MIN_SAMPLES = _env_int("OPTICS_MIN_SAMPLES", 3)
OPTICS_XI = _env_float("OPTICS_XI", 0.05)
OPTICS_MIN_CLUSTER_SIZE = None
NOISE_ASSIGNMENT_THRESHOLD = _env_float(
    "NOISE_ASSIGNMENT_THRESHOLD",
    0.75,
)

# Trust evaluation
MIN_REFERENCE_CLIENTS = _env_int("MIN_REFERENCE_CLIENTS", 3)

# Main / backup client selection
SELECTION_ALPHA = _env_float("SELECTION_ALPHA", 0.85)
BACKUP_RATIO = _env_float("BACKUP_RATIO", 0.5)
RANDOM_RATIO = _env_float("RANDOM_RATIO", 0.65)

# Trust evaluation
T_NEAR = _env_float("T_NEAR", 0.85)
LAMBDA_TRUST = _env_float("LAMBDA_TRUST", 0.8)
TRUST_THRESHOLD = _env_float("TRUST_THRESHOLD", 0.35)
INITIAL_TRUST = _env_float("INITIAL_TRUST", 0.5)

# Robust intra-cluster aggregation
TRIM_RATIO = _env_float("TRIM_RATIO", 0.1)



# Convergence

# MODEL_CHANGE_THRESHOLD = _env_float(
#     "MODEL_CHANGE_THRESHOLD",
#     0.007,
# )

VAL_LOSS_CHANGE_THRESHOLD = _env_float(
    "VAL_LOSS_CHANGE_THRESHOLD",
    0.0095,
)

PATIENCE = _env_int("PATIENCE", 3)


# Multi-Krum baseline

MULTI_KRUM_F = _env_int("MULTI_KRUM_F", 2)

#ِDataset

DATASET = "CIFAR10"


if DATASET == "MNIST":

    LEARNING_RATE = 0.01
    MODEL_CHANGE_THRESHOLD = 0.007


elif DATASET == "CIFAR10":

    LEARNING_RATE = 0.01
    MODEL_CHANGE_THRESHOLD = 0.015




