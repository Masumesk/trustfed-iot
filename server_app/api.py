from fastapi import FastAPI
from pydantic import BaseModel
from federated.server import Server
from data.load_dataset import load_dataset

import torch
from config import (
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_REFERENCE_CLIENTS,
    MODEL_SEED,
    DATASET,
)

import numpy as np
import io
from fastapi import FastAPI, Request , Response


torch.manual_seed(MODEL_SEED)


# class ClientUpdate(BaseModel):
#     client_id: int
#     update: list

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

def build_training_package_cache():

    global _training_package_bytes
    global _training_package_round


    package = {
        "selected": True,

        "round":
            server.current_round,

        "global_model": {
            name:
                tensor.detach().cpu()

            for name, tensor
            in server.global_model
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
        buffer,
    )


    _training_package_bytes = (
        buffer.getvalue()
    )

    _training_package_round = (
        server.current_round
    )


server = Server()
_, val_dataset, test_dataset = load_dataset(DATASET)
server.val_dataset = val_dataset
server.test_dataset = test_dataset

app = FastAPI(title="TrustFed IoT Server")
_training_package_bytes = None
_training_package_round = None


@app.post("/start_round")
def start_round(data: RoundRequest):

    global _training_package_bytes
    global _training_package_round


    server.start_round(
        data.round_id
    )

    _training_package_bytes = None
    _training_package_round = None


    return {
        "status": "round started",
        "round_id": data.round_id,
    }


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


@app.post("/initialize_clustering")
def initialize_clustering():

    if len(server.client_infos) < MIN_REFERENCE_CLIENTS:
        return {
            "status": "waiting",
            "registered": len(server.client_infos),
            "expected": MIN_REFERENCE_CLIENTS
        }

   
    server.receive_client_distributions(
        server.client_infos
    )

    server.client_clustering()

    return {
        "status": "clustering initialized",
        "clusters": server.clusters
    }


@app.post("/prepare_round")
def prepare_round():

    server.main_and_backup_client_selection()

    build_training_package_cache()


    return {
        "status": "round prepared",
        "clusters": server.clusters,
        "main_clients":
            server.main_clients,
        "backup_clients":
            server.backup_clients,
    }

@app.get("/model/{client_id}")
def get_model(client_id: int):

    selected = (
        get_selected_client_ids(
            server
        )
    )


    if client_id not in selected:

        return {
            "status":
                "not_selected",

            "selected":
                False,

            "round":
                server.current_round,

            "client_id":
                client_id,
        }


    if (
        _training_package_bytes
        is None
        or
        _training_package_round
        != server.current_round
    ):

        raise RuntimeError(
            "Training package cache "
            "is not initialized for "
            "the current round."
        )


    return Response(
        content=
            _training_package_bytes,

        media_type=
            "application/octet-stream",
    )

@app.post("/update")
async def receive_update(
    client_id: int,
    request: Request,
    round_id: int,
):
    if round_id != server.current_round: #check to ensure the update is for current round
        return {
            "status":
                "rejected_stale_round",

            "client_id":
                client_id,

            "received_round":
                round_id,

            "current_round":
                server.current_round,
        }

    selected = get_selected_client_ids(
        server
    )

    all_backup_ids = set()

    for clients in (
        server.backup_clients.values()
    ):
        all_backup_ids.update(clients)

    if (
        client_id not in selected
        and client_id not in all_backup_ids
    ):
        return {
            "status":
                "rejected_not_selected",
            "client_id":
                client_id
        }

    body = await request.body()

    update = np.frombuffer(
        body,
        dtype=np.float32
    )

    server.receive_client_update(
        client_id,
        update
    )

    return {
        "status": "update stored",
        "client_id": client_id
    }


@app.post("/aggregate")
def aggregate():

    if server.round_aggregated:

        return {
            "status":
                "aggregation completed",

            "accepted_clients":
                server.accepted_clients,

            "trust_scores":
                server.trust_scores,

            "model_relative_change":
                server.model_relative_change,
        }


    
    # Phase 1:
    # Reference 

    if not server.main_trust_evaluated:

        reference_requirements = (
            server
            .prepare_reference_backup_request()
        )


        if reference_requirements:

            return {
                "status":
                    "backup_needed",

                "stage":
                    "reference",

                "backup_requirements":
                    reference_requirements,
            }


        # Reference 
        # Main trust is updated
       
        server.evaluate_main_trust_once()


    
    # Phase 2:
    # Suspicious-main backups
    replacement_requirements = (
        server.prepare_replacement_backup_request()
    )


    if replacement_requirements:

        return {
            "status":
                "backup_needed",

            "stage":
                "replacement",

            "backup_requirements":
                replacement_requirements,
        }


    
    # Phase 3:
    # Evaluate backups,
    # replacement
    server.finalize_backup_replacement()


    # Phase 4:
    # Final aggregation

    server.aggregate()


    return {
        "status":
            "aggregation completed",

        "accepted_clients":
            server.accepted_clients,

        "trust_scores":
            server.trust_scores,

        "model_relative_change":
            server.model_relative_change,
    }



@app.get("/evaluate")
def evaluate(use_val: bool = True):
    return server.evaluate(use_val=use_val)


@app.get("/evaluate_final")
def evaluate_final():
    """Final evaluation on test set (use only after parameter selection)."""
    return server.evaluate(use_val=False)


@app.get("/round_selection")
def round_selection():

    fairness = (
        server.get_representation_fairness()
    )

    return {
        "main_clients":
            server.main_clients,

        "backup_clients":
            server.backup_clients,

        "representation_fairness":
            fairness["fairness"],

        "hellinger_distance":
            fairness["hellinger_distance"],

        "global_distribution":
            fairness["global_distribution"],

        "selected_distribution":
            fairness["selected_distribution"],
    }