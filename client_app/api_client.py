import requests
import torch
from models.model import MNISTCNN
from config import SERVER_URL as _DEFAULT_SERVER_URL


SERVER_URL = _DEFAULT_SERVER_URL


def set_server_url(url):
    global SERVER_URL
    SERVER_URL = url.rstrip("/")

def register_client(client_info):

    response = requests.post(
        f"{SERVER_URL}/register",
        json=client_info
    )

    response.raise_for_status()

    return response.json()


def get_training_package(client_id):

    response = requests.get(
        f"{SERVER_URL}/model/{client_id}"
    )

    response.raise_for_status()

    return response.json()


def load_global_model(package):

    model = MNISTCNN()

    state_dict = {
        k: torch.tensor(v)
        for k, v in package["global_model"].items()
    }

    model.load_state_dict(state_dict)

    return model


def send_update(
    client_id,
    update,
    round_id=None
):

    payload = {
        "client_id": client_id,
        "update": update.tolist()
    }

    if round_id is not None:
        payload["round_id"] = round_id

    response = requests.post(
        f"{SERVER_URL}/update",
        json=payload
    )

    response.raise_for_status()

    return response.json()