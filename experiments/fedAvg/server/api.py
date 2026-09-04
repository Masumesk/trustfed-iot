import random

import numpy as np
import torch
import io

from fastapi import (
    FastAPI,
    Request,
    Response,
)

from pydantic import BaseModel

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_SEED,
    NUM_CLIENTS,
    PARTICIPATION_RATIO,
    SELECTION_SEED,
    DATASET,
)
from data.load_dataset import load_dataset
from evaluation.global_evaluate import evaluate_model
from experiments.fedAvg.fedavg import fedavg_updates
from evaluation.evaluate_updates.model_update import apply_model_update
from models.get_model import get_model

app = FastAPI()

# Server state

clients = {}
selected_clients = []
updates = {}

current_round = 0

torch.manual_seed(MODEL_SEED)
global_model = get_model()
_model_state_cache = None

_, val_dataset, test_dataset = load_dataset(DATASET)


NUM_SELECTED = round(
    NUM_CLIENTS * PARTICIPATION_RATIO
)


# Request models

class ClientInfo(BaseModel):
    client_id: int
    distribution: list
    num_samples: int


# class ClientUpdate(BaseModel):
#     client_id: int
#     update: list[float]


# API

@app.get("/")
def root():
    return {
        "server": "FedAvg",
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
    global _model_state_cache

    if len(clients) < NUM_SELECTED:
        raise ValueError(
            f"Not enough registered clients: "
            f"registered={len(clients)}, required={NUM_SELECTED}"
        )

    current_round += 1
    _model_state_cache = None
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

    global _model_state_cache

    if _model_state_cache is None:

        package = {
            "selected": True,

            "round":
                current_round,

            "global_model": {
                name:
                    tensor.detach().cpu()

                for name, tensor
                in global_model
                    .state_dict()
                    .items()
            },

            "local_epochs":
                LOCAL_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,
        }

        buffer = io.BytesIO()

        torch.save(
            package,
            buffer
        )

        _model_state_cache = (
            buffer.getvalue()
        )

    return Response(
        content=_model_state_cache,
        media_type=
            "application/octet-stream"
    )


@app.post("/update")
async def receive_update(
    client_id: int,
    request: Request,
):

    if client_id not in selected_clients:

        raise ValueError(
            f"Client {client_id} "
            f"is not selected "
            f"in round {current_round}."
        )

    body = await request.body()

    
    received_update = np.frombuffer(
        body,
        dtype=np.float64
    )

    updates[client_id] = received_update.copy()

    return {
        "received": len(updates),
        "expected":
            len(selected_clients),
    }


@app.post("/aggregate")
def aggregate():

    global _model_state_cache

    missing_clients = sorted(
        set(selected_clients) - set(updates.keys())
    )

    if missing_clients:
        raise ValueError(
            f"Missing updates from clients: {missing_clients}"
        )

    round_updates = [
        updates[client_id]
        for client_id in selected_clients
    ]

    sample_counts = [
        clients[client_id].num_samples
        for client_id in selected_clients
    ]

    aggregated_update = fedavg_updates(
        round_updates,
        sample_counts,
    )

    if not np.isfinite(aggregated_update).all():
        bad_count = int(
            np.size(aggregated_update)
            - np.count_nonzero(np.isfinite(aggregated_update))
        )

        print(
            f"[WARNING] Round {current_round} | "
            f"aggregated update contains "
            f"{bad_count} non-finite values.",
            flush=True,
        )

        return {
            "round": current_round,
            "relative_change": 1e12,
            "diverged": True,
            "reason": "non_finite_aggregated_update",
        }

    
    with torch.no_grad():
        model_sq_sum = sum(
            torch.sum(
                parameter.detach().double() ** 2
            ).item()
            for parameter in global_model.parameters()
        )

    model_norm = float(np.sqrt(model_sq_sum))

    update_norm = float(
        np.linalg.norm(
            np.asarray(
                aggregated_update,
                dtype=np.float64,
            )
        )
    )

    relative_change = (
        update_norm / (model_norm + 1e-12)
    )

    if not np.isfinite(relative_change):
        print(
            f"[WARNING] Round {current_round} | "
            f"relative_change became {relative_change}.",
            flush=True,
        )

        return {
            "round": current_round,
            "relative_change": 1e12,
            "diverged": True,
            "reason": "non_finite_relative_change",
        }

    apply_model_update(
        global_model,
        aggregated_update,
    )

    _model_state_cache = None

    

    return {
        "round": current_round,
        "relative_change": float(relative_change),
    }


@app.get("/evaluate")
def evaluate():
    result = evaluate_model(
        global_model,
        val_dataset,
    )
    return result

@app.get("/evaluate_final")
def evaluate_final():
    result = evaluate_model(
        global_model,
        test_dataset,
    )
    return result