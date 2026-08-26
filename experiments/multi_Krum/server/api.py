import random

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_SEED,
    MULTI_KRUM_F,
    NUM_CLIENTS,
    PARTICIPATION_RATIO,
    SELECTION_SEED,
)
from data import load_mnist
from evaluation.global_evaluate import evaluate_model
from experiments.multi_Krum.multi_Krum import multi_krum
from evaluation.evaluate_updates.model_update import apply_model_update
from models.model import MNISTCNN

app = FastAPI()

# Server state

clients = {}
selected_clients = []
updates = {}

current_round = 0

torch.manual_seed(MODEL_SEED)
global_model = MNISTCNN()

_, test_dataset = load_mnist()

NUM_SELECTED = round(
    NUM_CLIENTS * PARTICIPATION_RATIO
)


# Request models

class ClientInfo(BaseModel):
    client_id: int
    distribution: list
    num_samples: int


class ClientUpdate(BaseModel):
    client_id: int
    update: list[float]


# API

@app.get("/")
def root():
    return {
        "server": "Multi-Krum",
        "round": current_round,
    }


@app.post("/register")
def register(info: ClientInfo):

    clients[info.client_id] = info

    return {
        "client_id": info.client_id,
        "registered": len(clients),
    }


@app.post("/start_round")
def start_round():

    global current_round
    global selected_clients
    global updates

    if len(clients) < NUM_SELECTED:
        raise ValueError(
            f"Not enough registered clients: "
            f"registered={len(clients)}, required={NUM_SELECTED}"
        )

    current_round += 1
    updates = {}

    # Random selection remains independent of malicious identity.
    rng = random.Random(
        SELECTION_SEED + current_round
    )

    selected_clients = rng.sample(
        sorted(clients.keys()),
        NUM_SELECTED,
    )

    return {
        "round": current_round,
        "selected_clients": selected_clients,
    }


@app.get("/model/{client_id}")
def get_model(client_id: int):

    if client_id not in selected_clients:
        return {
            "selected": False,
            "round": current_round,
        }

    model = {
        name: value.tolist()
        for name, value in global_model.state_dict().items()
    }

    return {
        "selected": True,
        "round": current_round,
        "global_model": model,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }


@app.post("/update")
def receive_update(data: ClientUpdate):

    if data.client_id not in selected_clients:
        raise ValueError(
            f"Client {data.client_id} is not selected "
            f"in round {current_round}."
        )

    updates[data.client_id] = np.array(
        data.update,
        dtype=np.float32,
    )

    return {
        "received": len(updates),
        "expected": len(selected_clients),
    }


@app.post("/aggregate")
def aggregate():

    missing_clients = sorted(
        set(selected_clients) - set(updates.keys())
    )

    if missing_clients:
        raise ValueError(
            f"Missing updates from clients: {missing_clients}"
        )

    aggregated_update, krum_selected, scores = multi_krum(
        updates,
        f=MULTI_KRUM_F,
    )

    # Same relative-change calculation as the standalone baseline:
    with torch.no_grad():
        model_norm = torch.sqrt(
            sum(
                torch.sum(parameter.detach() ** 2)
                for parameter in global_model.parameters()
            )
        ).item()

    relative_change = (
        np.linalg.norm(aggregated_update)
        / (model_norm + 1e-12)
    )

    apply_model_update(
        global_model,
        aggregated_update,
    )

    return {
        "round": current_round,
        "selected_by_krum": krum_selected,
        "scores": {
            int(client_id): float(score)
            for client_id, score in scores.items()
        },
        "relative_change": float(relative_change),
    }


@app.get("/evaluate")
def evaluate_global_model():

    result = evaluate_model(
        global_model,
        test_dataset,
    )

    return {
        "round": current_round,
        "accuracy": float(result["accuracy"]),
        "loss": float(result["loss"]),
    }
