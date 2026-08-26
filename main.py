import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from evaluation.csv_output import save_round_to_csv

from config import (MODEL_CHANGE_THRESHOLD, NUM_ROUNDS, PATIENCE,
                    VAL_LOSS_CHANGE_THRESHOLD, get_malicious_ids)

SERVER = "http://127.0.0.1:8000"


previous_val_loss = None

stable_checks = 0


malicious_ids = get_malicious_ids()


print(
    "Malicious clients:",
    sorted(malicious_ids)
)


# Run one selected client

def run_client(client_id):

    print(
        f"\nStarting client {client_id}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "client_app.run_round",
            "--id",
            str(client_id)
        ]
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Client {client_id} failed"
        )

    return client_id



# Federated rounds

for round_id in range(
    1,
    NUM_ROUNDS + 1
):

    print("\n========================")
    print(f"ROUND {round_id}")
    print("========================")

     # Start round
    response = requests.post(
        f"{SERVER}/start_round",
        json={
            "round_id": round_id
        }
    )

    response.raise_for_status()

    start_result = response.json()

    print(
        "Round started:",
        start_result
    )


    # Prepare round

    response = requests.post(
        f"{SERVER}/prepare_round"
    )

    response.raise_for_status()

    prepare_result = response.json()


    if (
        prepare_result.get("status")
        == "waiting"
    ):

        print(
            "Not enough registered clients:",
            prepare_result
        )

        break


    # Selected clients

    response = requests.get(
        f"{SERVER}/round_selection"
    )

    response.raise_for_status()

    selection = response.json()


    main_clients = selection[
        "main_clients"
    ]

    backup_clients = selection[
        "backup_clients"
    ]


    main_ids = []

    backup_ids = []


    for cluster_clients in (
        main_clients.values()
    ):

        main_ids.extend(
            cluster_clients
        )


    for cluster_clients in (
        backup_clients.values()
    ):

        backup_ids.extend(
            cluster_clients
        )


    selected_clients = sorted(
        set(
            main_ids
            +
            backup_ids
        )
    )


    print(
        "Main clients:",
        main_ids
    )

    print(
        "Backup clients:",
        backup_ids
    )

    print(
        "Selected malicious main clients:",
        sorted(
            set(main_ids)
            &
            malicious_ids
        )
    )

    print(
        "Selected malicious backup clients:",
        sorted(
            set(backup_ids)
            &
            malicious_ids
        )
    )

    # Parallel local training


    with ThreadPoolExecutor(
        max_workers=len(
            selected_clients
        )
    ) as executor:

        futures = [
            executor.submit(
                run_client,
                client_id
            )
            for client_id
            in selected_clients
        ]


        for future in futures:

            future.result()


    print(
        "\nAll selected clients finished"
    )

    # Trust + aggregation


    response = requests.post(
        f"{SERVER}/aggregate"
    )

    response.raise_for_status()

    aggregation_result = (
        response.json()
    )


    print(
        "Accepted clients:",
        aggregation_result[
            "accepted_clients"
        ]
    )

    print(
        "Trust scores:",
        aggregation_result[
            "trust_scores"
        ]
    )


    model_relative_change = (
        aggregation_result[
            "model_relative_change"
        ]
    )


    print(
        "Model relative change:",
        model_relative_change
    )

    # Global evaluation


    response = requests.get(
        f"{SERVER}/evaluate"
    )

    response.raise_for_status()

    evaluation = response.json()


    val_accuracy = (
        evaluation["accuracy"]
    )

    val_loss = (
        evaluation["loss"]
    )

    selected_malicious = sorted(
        set(selected_clients)
        &
        malicious_ids
    )

    accepted_ids = []

    for cluster_clients in aggregation_result[
        "accepted_clients"
    ].values():

        for item in cluster_clients:

            if isinstance(item, (list, tuple)):
                client_id = item[0]
            else:
                client_id = item

            accepted_ids.append(
                int(client_id)
            )

    malicious_kept = [
        client_id
        for client_id in selected_malicious
            if client_id in accepted_ids
    ]

    malicious_rejected = [
        client_id
        for client_id in selected_malicious
            if client_id not in accepted_ids
    ]

    save_round_to_csv(
        "results/proposed.csv",
        {
            "round": round_id,
            "accuracy": val_accuracy,
            "loss": val_loss,
            "relative_change": model_relative_change,
            "selected_malicious": len(
                selected_malicious
            ),
            "malicious_kept": len(
                malicious_kept
            ),
            "malicious_rejected": len(
                malicious_rejected
            ),
        }
    )


    print(
        f"Global Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Global Loss: "
        f"{val_loss:.6f}"
    )


    # Stopping criterion

    if (
        model_relative_change
        <
        MODEL_CHANGE_THRESHOLD
    ):

        if previous_val_loss is not None:

            val_loss_change = abs(
                val_loss
                -
                previous_val_loss
            )


            print(
                "Validation loss change:",
                val_loss_change
            )


            if (
                val_loss_change
                <
                VAL_LOSS_CHANGE_THRESHOLD
            ):

                stable_checks += 1

            else:

                stable_checks = 0


        previous_val_loss = (
            val_loss
        )


        print(
            f"Stable checks: "
            f"{stable_checks}/{PATIENCE}"
        )


        if (
            stable_checks
            >=
            PATIENCE
        ):

            print(
                "\nTraining stopped:"
            )

            print(
                "Convergence criteria "
                "satisfied."
            )

            break


    else:

        stable_checks = 0

        previous_val_loss = None


    print(
        f"ROUND {round_id} completed."
    )

    time.sleep(1)