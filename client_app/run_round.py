import argparse
import pickle
import random

import numpy as np
import torch

from attacks.attack_manager import apply_attack
from client_app.api_client import get_training_package, send_update, set_server_url
from config import ATTACK_TYPE, DATASET, SERVER_URL, TRAINING_SEED, get_malicious_ids
from data.load_dataset import load_dataset
from evaluation.evaluate_updates.model_update import (
    compute_model_update_from_state_dict,
    state_dict_to_parameter_vector,
)
from federated.client import Client
from models.get_model import get_model

_TRAIN_DATASET = None
_CLIENT_INDICES = None
_CLIENT_CACHE = {}
_MALICIOUS_IDS = None
_WORKER_LOCAL_MODEL = None

# for caching global model
_CACHED_TRAINING_PACKAGE = None
_CACHED_PACKAGE_ROUND = None
_CACHED_PACKAGE_SERVER_URL = None

# Cache global state on the active device
_CACHED_GLOBAL_STATE_DEVICE = None
_CACHED_GLOBAL_STATE_KEY = None
_CACHED_GLOBAL_VECTOR = None
_CACHED_GLOBAL_VECTOR_KEY = None


def _ensure_worker_data():

    global _TRAIN_DATASET
    global _CLIENT_INDICES
    global _MALICIOUS_IDS

    if _TRAIN_DATASET is None:
        _TRAIN_DATASET, _, _ = load_dataset(DATASET, load_test=False)

    if _CLIENT_INDICES is None:
        with open("data/partition_cache.pkl", "rb") as f:
            _CLIENT_INDICES = pickle.load(f)

    if _MALICIOUS_IDS is None:
        _MALICIOUS_IDS = get_malicious_ids()


def _get_client(client_id):

    _ensure_worker_data()

    if client_id not in _CLIENT_CACHE:

        is_malicious = client_id in _MALICIOUS_IDS

        _CLIENT_CACHE[client_id] = Client(
            client_id=client_id,
            dataset=_TRAIN_DATASET,
            indices=_CLIENT_INDICES[client_id],
            num_classes=10,
            malicious=is_malicious,
            attack_type=ATTACK_TYPE,
            compute_distribution=False,
        )

    return _CLIENT_CACHE[client_id]


def _get_cached_training_package(
    client_id,
    server_url,
    expected_round=None,
):
    global _CACHED_TRAINING_PACKAGE
    global _CACHED_PACKAGE_ROUND
    global _CACHED_PACKAGE_SERVER_URL

    normalized_server_url = server_url.rstrip("/")

    if expected_round is None:
        return get_training_package(client_id)

    expected_round = int(expected_round)

    if (
        _CACHED_TRAINING_PACKAGE is not None
        and _CACHED_PACKAGE_ROUND == expected_round
        and _CACHED_PACKAGE_SERVER_URL == normalized_server_url
    ):
        return _CACHED_TRAINING_PACKAGE

    package = get_training_package(client_id)

    if package.get("selected") is False:
        return package

    package_round = int(
        package.get(
            "round",
            0,
        )
    )

    if package_round != expected_round:
        raise RuntimeError(
            "Training package round mismatch: "
            f"expected={expected_round}, "
            f"received={package_round}"
        )

    _CACHED_TRAINING_PACKAGE = package
    _CACHED_PACKAGE_ROUND = package_round
    _CACHED_PACKAGE_SERVER_URL = normalized_server_url

    return _CACHED_TRAINING_PACKAGE


def run_client_round(
    client_id,
    server_url=SERVER_URL,
    send_immediately=True,
    expected_round=None,
):
    global _WORKER_LOCAL_MODEL
    set_server_url(server_url)
    client = _get_client(client_id)
    is_malicious = client.malicious

    package = _get_cached_training_package(
        client_id=client_id,
        server_url=server_url,
        expected_round=expected_round,
    )

    if package.get("selected") is False:

        print(
            f"Client {client_id} was not "
            f"selected in round "
            f"{package.get('round')}"
        )

        return {
            "client_id": client_id,
            "round_id": int(
                package.get(
                    "round",
                    0,
                )
            ),
            "sent": False,
            "update": None,
            "selected": False,
        }

    round_id = int(package.get("round", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _WORKER_LOCAL_MODEL is None:
        _WORKER_LOCAL_MODEL = get_model().to(device)

    global_state_device = _get_cached_global_state_on_device(
        package=package,
        server_url=server_url,
        round_id=round_id,
        device=device,
    )

    global_vector = _get_cached_global_vector(
        global_state_dict=global_state_device,
        local_model=_WORKER_LOCAL_MODEL,
        server_url=server_url,
        round_id=round_id,
        device=device,
    )

    training_seed = TRAINING_SEED + 10000 * round_id + client_id

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)

    local_model, loss = client.local_train(
        epochs=package["local_epochs"],
        batch_size=package["batch_size"],
        lr=package["learning_rate"],
        reusable_model=_WORKER_LOCAL_MODEL,
        global_state_dict=global_state_device,
        track_loss=False,
    )

    update = compute_model_update_from_state_dict(
        global_state_device, local_model, global_vector=global_vector
    )

    if is_malicious and ATTACK_TYPE is not None and ATTACK_TYPE != "label_flip":
        update = apply_attack(
            update,
            ATTACK_TYPE,
        )

    if send_immediately:
        response = send_update(client_id, update, round_id=round_id)

        return {
            "client_id": client_id,
            "round_id": round_id,
            "sent": True,
            "update": None,
        }

    return {
        "client_id": client_id,
        "round_id": round_id,
        "sent": False,
        "update": update,
    }


def _get_cached_global_state_on_device(
    package,
    server_url,
    round_id,
    device,
):
    global _CACHED_GLOBAL_STATE_DEVICE
    global _CACHED_GLOBAL_STATE_KEY

    cache_key = (
        server_url.rstrip("/"),
        int(round_id),
        str(device),
    )

    if (
        _CACHED_GLOBAL_STATE_DEVICE is not None
        and _CACHED_GLOBAL_STATE_KEY == cache_key
    ):
        return _CACHED_GLOBAL_STATE_DEVICE

    global_state = package["global_model"]

    if device.type == "cpu":
        _CACHED_GLOBAL_STATE_DEVICE = global_state

    else:
        _CACHED_GLOBAL_STATE_DEVICE = {
            name: tensor.detach().to(device) for name, tensor in global_state.items()
        }

    _CACHED_GLOBAL_STATE_KEY = cache_key

    return _CACHED_GLOBAL_STATE_DEVICE


def _get_cached_global_vector(
    global_state_dict,
    local_model,
    server_url,
    round_id,
    device,
):
    global _CACHED_GLOBAL_VECTOR
    global _CACHED_GLOBAL_VECTOR_KEY

    cache_key = (
        server_url.rstrip("/"),
        int(round_id),
        str(device),
    )

    if _CACHED_GLOBAL_VECTOR is not None and _CACHED_GLOBAL_VECTOR_KEY == cache_key:
        return _CACHED_GLOBAL_VECTOR

    _CACHED_GLOBAL_VECTOR = state_dict_to_parameter_vector(
        global_state_dict,
        local_model,
    )

    _CACHED_GLOBAL_VECTOR_KEY = cache_key

    return _CACHED_GLOBAL_VECTOR


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--server",
        type=str,
        default=SERVER_URL,
    )

    args = parser.parse_args()

    run_client_round(
        args.id,
        args.server,
    )


if __name__ == "__main__":
    main()
