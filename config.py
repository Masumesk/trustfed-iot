import numpy as np


# Federated configuration

NUM_CLIENTS = 30

NUM_ROUNDS = 50

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Data distribution


DIRICHLET_ALPHA = 0.3
MIN_SAMPLES = 100
DATA_SEED = 42

# Attack configuration

ATTACK_TYPE = "label_flip"

MALICIOUS_RATIO = 0.2
MALICIOUS_SEED = 42


def get_malicious_ids():

    rng = np.random.RandomState(
        MALICIOUS_SEED
    )

    num_malicious = int(
        NUM_CLIENTS * MALICIOUS_RATIO
    )

    malicious_ids = rng.choice(
        NUM_CLIENTS,
        num_malicious,
        replace=False
    )

    return set(
        int(client_id)
        for client_id in malicious_ids
    )

# Convergence

MODEL_CHANGE_THRESHOLD = 0.01

VAL_LOSS_CHANGE_THRESHOLD = 0.01

PATIENCE = 3