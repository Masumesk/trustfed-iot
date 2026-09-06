import io
import random

import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from config import (
    BATCH_SIZE,
    DATASET,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_SEED,
    NUM_CLIENTS,
    PARTICIPATION_RATIO,
    SELECTION_SEED,
)
from data.load_dataset import load_dataset
from evaluation.global_evaluate import evaluate_model
from models.get_model import get_model


class ClientInfo(BaseModel):

    client_id: int
    distribution: list
    num_samples: int


def create_baseline_app(
    *,
    server_name,
    aggregate_round,
    copy_received_update,
):

    app = FastAPI()
    clients = {}
    selected_clients = []
    updates = {}
    current_round = 0
    torch.manual_seed(MODEL_SEED)
    global_model = get_model()
    model_state_cache = None
    _, val_dataset, test_dataset = load_dataset(DATASET)
    num_selected = round(NUM_CLIENTS * PARTICIPATION_RATIO)

    @app.get("/")
    def root():
        return {
            "server": server_name,
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
        nonlocal current_round
        nonlocal selected_clients
        nonlocal updates
        nonlocal model_state_cache

        if len(clients) < num_selected:

            raise ValueError("Not enough registered clients: "f"registered={len(clients)}, "f"required={num_selected}")

        current_round += 1
        model_state_cache = None
        updates = {}
        rng = random.Random(SELECTION_SEED + current_round)

        selected_clients = rng.sample(
            sorted(clients.keys()),
            num_selected,
        )

        return {
            "round": current_round,
            "selected_clients": selected_clients,
        }

    @app.get("/model/{client_id}")
    def get_training_model(
        client_id: int,
    ):
        nonlocal model_state_cache

        if client_id not in selected_clients:

            return {
                "selected": False,
                "round": current_round,
            }

        if model_state_cache is None:

            package = {
                "selected": True,
                "round": current_round,
                "global_model": {
                    name: tensor.detach().cpu()
                    for name, tensor in global_model.state_dict().items()
                },
                "local_epochs": LOCAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            }

            buffer = io.BytesIO()
            torch.save(package, buffer)
            model_state_cache = buffer.getvalue()

        return Response(
            content=model_state_cache,
            media_type="application/octet-stream",
        )

    @app.post("/update")
    async def receive_update(
        client_id: int,
        request: Request,
    ):
        if client_id not in selected_clients:
            raise ValueError(f"Client {client_id} is not selected " f"in round {current_round}.")

        body = await request.body()

        received_update = np.frombuffer(
            body,
            dtype=np.float32,
        )

        if copy_received_update:
            received_update = received_update.copy()

        updates[client_id] = received_update

        return {
            "received": len(updates),
            "expected": len(selected_clients),
        }

    @app.post("/aggregate")
    def aggregate():

        nonlocal model_state_cache
        missing_clients = sorted(set(selected_clients) - set(updates.keys()))

        if missing_clients:
            raise ValueError("Missing updates from clients: " f"{missing_clients}")

        result, model_changed = aggregate_round(
            global_model=global_model,
            clients=clients,
            selected_clients=selected_clients,
            updates=updates,
            current_round=current_round,
        )

        if model_changed:
            model_state_cache = None

        return result

    @app.get("/evaluate")
    def evaluate():

        return evaluate_model(
            global_model,
            val_dataset,
        )

    @app.get("/evaluate_final")
    def evaluate_final():

        return evaluate_model(
            global_model,
            test_dataset,
        )

    return app
