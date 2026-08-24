from fastapi import FastAPI
from pydantic import BaseModel
from federated.server import Server
from data import load_mnist

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

server = Server()

_, test_dataset = load_mnist()

server.test_dataset = test_dataset
app = FastAPI(
    title="TrustFed IoT Server"
)

@app.post("/start_round")
def start_round(data: RoundRequest):

    server.start_round(data.round_id)

    return {
        "status": "round started",
        "round_id": data.round_id
    }


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "TrustFed Server"
    }

@app.post("/register")
def register_client(info: ClientInfo):

    client_info = {
        "client_id": info.client_id,
        "distribution": info.distribution,
        "num_samples": info.num_samples
    }

    server.client_infos[info.client_id] = client_info

    return {
        "status": "registered",
        "client_id": info.client_id
    }


@app.post("/prepare_round")
def prepare_round():
    #  server.expected_clients
    if len(server.client_infos) < 3:
        return {
            "status": "waiting",
            "registered": len(server.client_infos),
            "expected": 3
        }


    server.receive_client_distributions(
        server.client_infos
    )

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

    return {
        "client_id": client_id,
        "global_model": server.get_model_state(),
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01
    }

@app.post("/update")
def receive_update(data: ClientUpdate):

    server.receive_client_update(
        data.client_id,
        np.array(data.update)
    )

    print("MAIN:", server.main_updates.keys())
    print("BACKUP:", server.backup_updates.keys())
    return {
        "status": "update received",
        "client_id": data.client_id
    }




@app.post("/aggregate")
def aggregate():

    server.trust_evaluation_and_backup_replacement()
    print(
        "MAIN UPDATES:",
        server.main_updates.keys()
    )
    print(
        "BACKUP UPDATES:",
        server.backup_updates.keys()
    )
    print(
        "ACCEPTED CLIENTS:",
        server.accepted_clients
    )
    print(
        "TRUST SCORES:",
        server.trust_scores
    )
    server.aggregate()
    return {
        "status": "aggregation completed",

        "accepted_clients":
            server.accepted_clients,

        "trust_scores":
            server.trust_scores,

        "model_relative_change":
            server.model_relative_change
    }


@app.get("/evaluate")
def evaluate():

    result = server.evaluate()

    return result


@app.get("/round_selection")
def round_selection():

    return {
        "main_clients": server.main_clients,
        "backup_clients": server.backup_clients
    }