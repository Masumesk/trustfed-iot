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

# Federated training
NUM_CLIENTS = _env_int("NUM_CLIENTS", 100)
PARTICIPATION_RATIO = _env_float("PARTICIPATION_RATIO", 0.1) 
NUM_ROUNDS = _env_int("NUM_ROUNDS", 1000)

LOCAL_EPOCHS = _env_int("LOCAL_EPOCHS", 2)
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
CLIENT_WORKERS = _env_int("CLIENT_WORKERS", 1)

# Reproducibility
MODEL_SEED = _env_int("MODEL_SEED", 42)
TRAINING_SEED = _env_int("TRAINING_SEED", 42)
SELECTION_SEED = _env_int("SELECTION_SEED", 42)

# Data partitioning
DIRICHLET_ALPHA = _env_float("DIRICHLET_ALPHA", 0.3)
MIN_SAMPLES = _env_int("MIN_SAMPLES", 100)
DATA_SEED = _env_int("DATA_SEED", 42)


# Attack configuration
ATTACK_TYPE = os.getenv("ATTACK_TYPE", None )
MALICIOUS_RATIO = _env_float("MALICIOUS_RATIO", 0.2)
MALICIOUS_SEED = _env_int("MALICIOUS_SEED", 42)


def get_malicious_ids():

    explicit_ids = os.getenv(
        "EXPERIMENT_MALICIOUS_IDS",
        "",
    ).strip()

    if explicit_ids:
        malicious_ids = {
            int(client_id)
            for client_id in explicit_ids.split(",")
            if client_id.strip()
        }

        invalid_ids = [
            client_id
            for client_id in malicious_ids
            if client_id < 0 or client_id >= NUM_CLIENTS
        ]

        if invalid_ids:
            raise ValueError(
                "Invalid malicious client IDs: "
                f"{sorted(invalid_ids)}"
            )

        return malicious_ids

    rng = np.random.RandomState(MALICIOUS_SEED)

    num_malicious = int(NUM_CLIENTS * MALICIOUS_RATIO)

    malicious_ids = rng.choice(
        NUM_CLIENTS,
        num_malicious,
        replace=False,
    )

    return {
        int(client_id)
        for client_id in malicious_ids
    }


# Clustering
OPTICS_MIN_SAMPLES = _env_int("OPTICS_MIN_SAMPLES", 3)
OPTICS_XI = _env_float("OPTICS_XI", 0.05)
OPTICS_MIN_CLUSTER_SIZE = None
NOISE_ASSIGNMENT_THRESHOLD = _env_float("NOISE_ASSIGNMENT_THRESHOLD", 0.75)

# Main / backup client selection
SELECTION_ALPHA = _env_float("SELECTION_ALPHA", 0.85)
BACKUP_RATIO = _env_float("BACKUP_RATIO", 0.5) 
RANDOM_RATIO = _env_float("RANDOM_RATIO", 0.65)

# Trust evaluation and suspicious-client handling
MIN_REFERENCE_CLIENTS = _env_int("MIN_REFERENCE_CLIENTS", 3) 
T_NEAR = _env_float("T_NEAR", 0.85)
LAMBDA_TRUST = _env_float("LAMBDA_TRUST", 0.8)
TRUST_THRESHOLD = _env_float("TRUST_THRESHOLD", 0.35) 
INITIAL_TRUST = _env_float("INITIAL_TRUST", 0.5)
SUSPICIOUS_PENALTY = _env_float("SUSPICIOUS_PENALTY", 0.3) ##

# Robust aggregation
TRIM_RATIO = _env_float("TRIM_RATIO", 0.1) 

# Convergence
VAL_LOSS_CHANGE_THRESHOLD = _env_float("VAL_LOSS_CHANGE_THRESHOLD", 0.0095)
PATIENCE = _env_int("PATIENCE", 3)

# Multi-Krum baseline
MULTI_KRUM_F = _env_int("MULTI_KRUM_F", 2)

# Dataset-specific settings
DATASET = "MNIST"

if DATASET == "MNIST":
    LEARNING_RATE = 0.01
    MODEL_CHANGE_THRESHOLD = 0.007

elif DATASET == "CIFAR10":
    LEARNING_RATE = 0.01
    MODEL_CHANGE_THRESHOLD = 0.015


# evaluation
EVAL_INTERVAL = 5
