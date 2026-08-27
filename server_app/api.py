from fastapi import FastAPI
from pydantic import BaseModel
from federated.server import Server
from data import load_mnist

from config import (
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_REFERENCE_CLIENTS,
)

import numpy as np


class ClientUpdate(BaseModel):
    client_id: int
    update: list

class RoundRequest(BaseModel):
    round_id: int

class ClientInfo(BaseModel):
    client_id: int
    distribution: list
    num_samples: int


def get_selected_client_ids(server):
    selected = set()
    for clients in server.main_clients.values():
        selected.update(clients)
    for clients in server.backup_clients.values():
        selected.update(clients)
    return selected


server = Server()
_, test_dataset = load_mnist()
server.test_dataset = test_dataset
app = FastAPI(title="TrustFed IoT Server")


@app.post("/start_round")
def start_round(data: RoundRequest):
    server.start_round(data.round_id)
    return {"status": "round started", "round_id": data.round_id}


@app.get("/")
def root():
    return {"status": "running", "service": "TrustFed Server"}


@app.post("/register")
def register_client(info: ClientInfo):
    client_info = {
        "client_id": info.client_id,
        "distribution": info.distribution,
        "num_samples": info.num_samples
    }
    server.client_infos[info.client_id] = client_info
    return {"status": "registered", "client_id": info.client_id}


@app.post("/prepare_round")
def prepare_round():
    if len(server.client_infos) < MIN_REFERENCE_CLIENTS:
        return {"status": "waiting", "registered": len(server.client_infos), "expected": MIN_REFERENCE_CLIENTS}

    server.receive_client_distributions(server.client_infos)
    server.client_clustering()
    server.main_and_backup_client_selection()

    return {
        "status": "round prepared",
        "clusters": server.clusters,
        "main_clients": server.main_clients,
        "backup_clients": server.backup_clients
    }


@app.get("/model/{client_id}")
def get_model(client_id: int):

    selected = get_selected_client_ids(server)

    if client_id not in selected:                     # ← چک جدید
        return {"status": "not_selected", "client_id": client_id}

    return {
        "client_id": client_id,
        "global_model": server.get_model_state(),
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }


@app.post("/update")
def receive_update(data: ClientUpdate):

    selected = get_selected_client_ids(server)

    # Also accept backup clients that were trained on demand
    all_backup_ids = set()
    for clients in server.backup_clients.values():
        all_backup_ids.update(clients)

    if (
        data.client_id not in selected
        and data.client_id not in all_backup_ids
    ):
        return {
            "status": "rejected_not_selected",
            "client_id": data.client_id
        }

    server.receive_client_update(data.client_id, np.array(data.update))

    return {"status": "update stored", "client_id": data.client_id}


@app.post("/aggregate")
def aggregate():
    server.trust_evaluation_and_backup_replacement()

    # Phase 1: backups needed — return requirements
    if server.backup_requirements:
        return {
            "status": "backup_needed",
            "backup_requirements": server.backup_requirements,
        }

    # Phase 2: all backups available — run aggregation
    server.aggregate()
    return {
        "status": "aggregation completed",
        "accepted_clients": server.accepted_clients,
        "trust_scores": server.trust_scores,
        "model_relative_change": server.model_relative_change,
    }


class BackupUpdate(BaseModel):
    client_id: int
    update: list


@app.post("/request_backup_updates")
def request_backup_updates(data: BackupUpdate):
    """
    Receive a single backup client's update on demand.
    """
    server.request_backup_updates(
        cluster_id=None,
        client_id=data.client_id,
        update=np.array(data.update),
    )
    return {
        "status": "backup_update_received",
        "client_id": data.client_id,
    }


@app.get("/evaluate")
def evaluate():
    return server.evaluate()


@app.get("/round_selection")
def round_selection():
    return {"main_clients": server.main_clients, "backup_clients": server.backup_clients}