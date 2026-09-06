import io

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter

from config import SERVER_URL as _DEFAULT_SERVER_URL

SERVER_URL = _DEFAULT_SERVER_URL.rstrip("/")

_SESSION = requests.Session()

_adapter = HTTPAdapter(
    pool_connections=8,
    pool_maxsize=8,
)

_SESSION.mount(
    "http://",
    _adapter,
)

_SESSION.mount(
    "https://",
    _adapter,
)


def set_server_url(url):
    global SERVER_URL
    SERVER_URL = url.rstrip("/")


def register_client(client_info):

    response = _SESSION.post(
        f"{SERVER_URL}/register",
        json=client_info,
    )
    response.raise_for_status()

    return response.json()


def get_training_package(client_id):

    response = _SESSION.get(f"{SERVER_URL}/model/{client_id}")
    response.raise_for_status()
    content_type = response.headers.get(
        "content-type",
        "",
    )

    if "application/json" in content_type:

        return response.json()

    return torch.load(
        io.BytesIO(response.content),
        map_location="cpu",
        weights_only=False,
    )


def send_update(
    client_id,
    update,
    round_id=None,
):

    update_to_send = np.ascontiguousarray(
        update,
        dtype=np.float32,
    )
    params = {"client_id": client_id}

    if round_id is not None:

        params["round_id"] = round_id

    response = _SESSION.post(
        f"{SERVER_URL}/update",
        params=params,
        data=update_to_send.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )

    response.raise_for_status()

    return response.json()
