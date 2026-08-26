import math
import random

import numpy as np
import torch

from attacks.attack_manager import apply_attack
from config import (ATTACK_TYPE, BATCH_SIZE, DATA_SEED, DIRICHLET_ALPHA,
                    LEARNING_RATE, LOCAL_EPOCHS, MALICIOUS_RATIO, MIN_SAMPLES,
                    MODEL_CHANGE_THRESHOLD, NUM_CLIENTS, NUM_ROUNDS, PATIENCE,
                    VAL_LOSS_CHANGE_THRESHOLD, get_malicious_ids)
from data import load_mnist
from data.partition import partition_dirichlet
from evaluate.global_evaluate import evaluate_model
from experiments.multi_Krum.multi_Krum import multi_krum
from federated.client import Client
from federated.model_update import apply_model_update, compute_model_update
from models.model import MNISTCNN

# config 

PARTICIPATION_RATIO = 0.3
MODEL_SEED = 42
TRAINING_SEED = 42
SELECTION_SEED = 42

n = round(NUM_CLIENTS * PARTICIPATION_RATIO)     # 9
F = math.ceil(n * MALICIOUS_RATIO)               # 2
M = n - F                                        # 7

if 2 * F + 2 >= n:
    raise ValueError("Multi-Krum condition: 2*f+2 < n")


# non-iid Data 

train_dataset, test_dataset = load_mnist()

client_indices = partition_dirichlet(
    train_dataset,
    NUM_CLIENTS,
    DIRICHLET_ALPHA,
    MIN_SAMPLES,
    DATA_SEED,
)

malicious_ids = get_malicious_ids()

benign_ids = [
    i for i in range(NUM_CLIENTS)
    if i not in malicious_ids
]

print("Malicious clients:", sorted(malicious_ids))
print(f"Multi-Krum: n={n}, f={F}, m={M}")


# model

torch.manual_seed(MODEL_SEED)
global_model = MNISTCNN()

previous_loss = None
stable_checks = 0


# train

for round_id in range(1, NUM_ROUNDS + 1):

    print(f"ROUND {round_id}")

    rng = random.Random(SELECTION_SEED + round_id)

    selected_malicious = rng.sample(
        sorted(malicious_ids),
        F
    )

    selected_benign = rng.sample(
        benign_ids,
        n - F
    )

    selected_clients = sorted(
        selected_malicious + selected_benign
    )

    print("Selected:", selected_clients)
    print("Malicious:", sorted(selected_malicious))

    client_updates = {}
    local_losses = []

    # local train
    for client_id in selected_clients:

        malicious = client_id in malicious_ids

        seed = (
            TRAINING_SEED
            + 10000 * round_id
            + client_id
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        client = Client(
            client_id=client_id,
            dataset=train_dataset,
            indices=client_indices[client_id],
            num_classes=10,
            malicious=malicious,
            attack_type=ATTACK_TYPE,
        )

        local_model, local_loss = client.local_train(
            global_model,
            LOCAL_EPOCHS,
            BATCH_SIZE,
            LEARNING_RATE,
        )

        update = compute_model_update(
            global_model,
            local_model
        )

        if malicious and ATTACK_TYPE not in (None, "label_flip"):
            update = apply_attack(
                update,
                ATTACK_TYPE
            )

        client_updates[client_id] = update
        local_losses.append(local_loss)

        print(
            f"Client {client_id:2d} | "
            f"{'MALICIOUS' if malicious else 'BENIGN':9s} | "
            f"loss={local_loss:.4f}"
        )


    # multi_Krum
    aggregated_update, krum_selected, scores = multi_krum(
        client_updates,
        f=F,
        m=M,
    )

    krum_selected = sorted(krum_selected)

    rejected = sorted(
        set(selected_clients)-set(krum_selected)
    )

    rejected_malicious = sorted(
        set(rejected) & malicious_ids
    )

    remaining_malicious = sorted(
        set(krum_selected) & malicious_ids
    )

    print("Multi-Krum scores:")

    for cid in sorted(scores, key=scores.get):
        print(
            f"{cid:2d}: {scores[cid]:.3f} | "
            f"{'M' if cid in malicious_ids else 'B'} | "
            f"{'KEEP' if cid in krum_selected else 'DROP'}"
        )


    # global update

    with torch.no_grad():
        model_norm = torch.sqrt(
            sum(
                torch.sum(p.detach() ** 2)
                for p in global_model.parameters()
            )
        ).item()

    relative_change = (
        np.linalg.norm(aggregated_update)/ (model_norm + 1e-12)
    )

    global_model = apply_model_update(
        global_model,
        aggregated_update
    )

    result = evaluate_model(
        global_model,
        test_dataset
    )

    accuracy = result["accuracy"]
    loss = result["loss"]


    # result
    print("results")
    print("rejected:", rejected)
    print("rejected malicious:", rejected_malicious)
    print("malicious still selected:", remaining_malicious)

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Loss: {loss:.4f} | "
        f"Change: {relative_change:.6f}"
    )


    # convergence 

    if (
        relative_change < MODEL_CHANGE_THRESHOLD
        and previous_loss is not None
        and abs(loss - previous_loss) < VAL_LOSS_CHANGE_THRESHOLD
    ):
        stable_checks += 1
    else:
        stable_checks = 0

    previous_loss = loss

    print(f"Stable: {stable_checks}/{PATIENCE}")

    if stable_checks >= PATIENCE:
        print(f"Converged at round {round_id}")
        break