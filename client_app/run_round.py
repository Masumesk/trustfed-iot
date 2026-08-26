import argparse
import random

import numpy as np
import torch

from attacks.attack_manager import apply_attack
from client_app.api_client import (
    get_training_package,
    load_global_model,
    send_update,
    set_server_url,
)
from config import (
    ATTACK_TYPE,
    DATA_SEED,
    DIRICHLET_ALPHA,
    MIN_SAMPLES,
    NUM_CLIENTS,
    TRAINING_SEED,
    get_malicious_ids,
)
from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client
from evaluation.evaluate_updates.model_update import compute_model_update


# Arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    "--id",
    type=int,
    required=True,
)

parser.add_argument(
    "--server",
    type=str,
    default="http://127.0.0.1:8000",
)

args = parser.parse_args()

client_id = args.id
set_server_url(args.server)


# Determine malicious state

malicious_ids = get_malicious_ids()
is_malicious = client_id in malicious_ids


# Recreate the same deterministic non-IID local partition

train_dataset, _ = load_mnist()

client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA,
    min_samples=MIN_SAMPLES,
    seed=DATA_SEED,
)

client = Client(
    client_id=client_id,
    dataset=train_dataset,
    indices=client_indices[client_id],
    num_classes=10,
    malicious=is_malicious,
    attack_type=ATTACK_TYPE,
)


# Receive current global model

package = get_training_package(
    client_id
)

if package.get("selected") is False:
    print(
        f"Client {client_id} was not selected "
        f"in round {package.get('round')}"
    )
    raise SystemExit(0)

round_id = int(
    package.get("round", 0)
)

global_model = load_global_model(
    package
)

print(
    f"Client {client_id} | "
    f"malicious={is_malicious} | "
    f"attack={ATTACK_TYPE}"
)
print("global model received")


# Deterministic local training seed for this (round, client)

training_seed = (
    TRAINING_SEED
    + 10000 * round_id
    + client_id
)

random.seed(training_seed)
np.random.seed(training_seed)
torch.manual_seed(training_seed)


# Local training

local_model, loss = client.local_train(
    model=global_model,
    epochs=package["local_epochs"],
    batch_size=package["batch_size"],
    lr=package["learning_rate"],
)

print(
    f"local loss: {loss}"
)

# Compute model update

update = compute_model_update(
    global_model,
    local_model,
)


# Apply model-update attack when appropriate

if (
    is_malicious
    and ATTACK_TYPE is not None
    and ATTACK_TYPE != "label_flip"
):

    update = apply_attack(
        update,
        ATTACK_TYPE,
    )

    print(
        f"ATTACK APPLIED | "
        f"client={client_id} | "
        f"type={ATTACK_TYPE}"
    )


# Send update

response = send_update(
    client_id,
    update,
)

print(response)
