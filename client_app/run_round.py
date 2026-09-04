import argparse
import pickle
import random

import numpy as np
import torch

from attacks.attack_manager import apply_attack

from client_app.api_client import (
    get_training_package,
    load_global_model,
    send_update,
    set_server_url,
)

from config import (
    ATTACK_TYPE,
    SERVER_URL,
    TRAINING_SEED,
    get_malicious_ids,
)

from models.get_model import get_model

from data.load_dataset import load_dataset

from evaluation.evaluate_updates.model_update import (
    compute_model_update,
)

from federated.client import Client

from config import (DATASET )




_TRAIN_DATASET = None
_CLIENT_INDICES = None
_CLIENT_CACHE = {}
_MALICIOUS_IDS = None
_WORKER_LOCAL_MODEL = None


def _ensure_worker_data():

    global _TRAIN_DATASET
    global _CLIENT_INDICES
    global _MALICIOUS_IDS

    if _TRAIN_DATASET is None:

        _TRAIN_DATASET, _, _ = (
            load_dataset(
                DATASET,
                load_test=False
            )
        )

    if _CLIENT_INDICES is None:

        with open(
            "data/partition_cache.pkl",
            "rb",
        ) as f:

            _CLIENT_INDICES = pickle.load(f)

    if _MALICIOUS_IDS is None:

        _MALICIOUS_IDS = get_malicious_ids()


def _get_client(client_id):

    _ensure_worker_data()

    if client_id not in _CLIENT_CACHE:

        is_malicious = (
            client_id in _MALICIOUS_IDS
        )

        _CLIENT_CACHE[client_id] = Client(
            client_id=client_id,
            dataset=_TRAIN_DATASET,
            indices=_CLIENT_INDICES[
                client_id
            ],
            num_classes=10,
            malicious=is_malicious,
            attack_type=ATTACK_TYPE,

            compute_distribution=False,
        )

    return _CLIENT_CACHE[client_id]



def run_client_round(
    client_id,
    server_url=SERVER_URL,
):

    global _WORKER_LOCAL_MODEL

    set_server_url(server_url)

    client = _get_client(client_id)

    is_malicious = client.malicious



    package = get_training_package(
        client_id
    )


    if package.get("selected") is False:

        print(
            f"Client {client_id} was not "
            f"selected in round "
            f"{package.get('round')}"
        )

        return client_id


    round_id = int(
        package.get(
            "round",
            0,
        )
    )


    global_model = load_global_model(
        package
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if _WORKER_LOCAL_MODEL is None:

        _WORKER_LOCAL_MODEL = (
            get_model()
            .to(device)
        )


    print(
        f"Client {client_id} | "
        f"malicious={is_malicious} | "
        f"attack={ATTACK_TYPE}"
    )

    print(
        "global model received"
    )


    

    training_seed = (
        TRAINING_SEED
        + 10000 * round_id
        + client_id
    )

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)



    local_model, loss = (
        client.local_train(
            model=global_model,
            epochs=package[
                "local_epochs"
            ],
            batch_size=package[
                "batch_size"
            ],
            lr=package[
                "learning_rate"
            ],
            reusable_model=
                _WORKER_LOCAL_MODEL,
        )
    )


    print(
        f"local loss: {loss}"
    )



    update = compute_model_update(
        global_model,
        local_model,
    )


    if (
        is_malicious
        and ATTACK_TYPE is not None
        and ATTACK_TYPE != "label_flip"
    ):

        update = apply_attack(
            update,
            ATTACK_TYPE,
        )

        print(
            f"ATTACK APPLIED | "
            f"client={client_id} | "
            f"type={ATTACK_TYPE}"
        )



    response = send_update(
        client_id,
        update,
    )

    print(response)

    return client_id


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