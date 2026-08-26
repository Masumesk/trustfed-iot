import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from config import (
    MODEL_CHANGE_THRESHOLD,
    NUM_ROUNDS,
    PATIENCE,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)



parser = argparse.ArgumentParser()

parser.add_argument(
    "--server",
    type=str,
    default="http://127.0.0.1:8001",
)

parser.add_argument(
    "--rounds",
    type=int,
    default=NUM_ROUNDS,
)

args = parser.parse_args()

server_url = args.server.rstrip("/")
num_rounds = args.rounds

malicious_ids = set(
    get_malicious_ids()
)


# Run one selected client

def run_client(client_id):

    print(
        f"\nRunning client {client_id}"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "client_app.run_round",
            "--id",
            str(client_id),
            "--server",
            server_url,
        ],
        check=True,
    )

    return client_id



def run_one_round(
    previous_loss,
    stable_checks,
):

    response = requests.post(
        f"{server_url}/start_round"
    )
    response.raise_for_status()

    round_info = response.json()
    round_id = round_info["round"]
    selected_clients = round_info["selected_clients"]

    selected_malicious = [
        client_id
        for client_id in selected_clients
        if client_id in malicious_ids
    ]

    print("\n" + "=" * 70)
    print(f"ROUND {round_id}")
    print("=" * 70)
    print(f"Selected clients:   {selected_clients}")
    print(f"Selected malicious: {selected_malicious}")

    # 2. Run selected clients
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
        "\nAll selected clients finished "
        "and sent their updates."
    )

    print("\nRunning FedAvg aggregation...")

    response = requests.post(
        f"{server_url}/aggregate"
    )
    response.raise_for_status()

    aggregation_result = response.json()

    print("\nAggregation result:")
    print(
        json.dumps(
            aggregation_result,
            indent=2,
        )
    )

    relative_change = float(
        aggregation_result["relative_change"]
    )

    malicious_kept = list(
        selected_malicious
    )

    malicious_rejected = []

    print("\nEvaluating global model...")

    response = requests.get(
        f"{server_url}/evaluate"
    )
    response.raise_for_status()

    evaluation = response.json()

    accuracy = float(
        evaluation["accuracy"]
    )
    loss = float(
        evaluation["loss"]
    )

    loss_change = None

    if previous_loss is not None:
        loss_change = abs(
            loss - previous_loss
        )

    if (
        relative_change < MODEL_CHANGE_THRESHOLD
        and loss_change is not None
        and loss_change < VAL_LOSS_CHANGE_THRESHOLD
    ):
        stable_checks += 1
    else:
        stable_checks = 0

    converged = (
        stable_checks >= PATIENCE
    )

    print("\n" + "-" * 70)
    print(f"ROUND {round_id} SUMMARY")
    print("-" * 70)
    print(f"Selected clients:       {selected_clients}")
    print(f"Selected malicious:     {selected_malicious}")
    print(f"Malicious aggregated:   {malicious_kept}")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Loss:                   {loss:.6f}")
    print(f"Relative change:        {relative_change:.6f}")

    if loss_change is None:
        print("Loss change:            N/A")
    else:
        print(f"Loss change:            {loss_change:.6f}")

    print(
        f"Stable:                 "
        f"{stable_checks}/{PATIENCE}"
    )
    print("-" * 70)

    return {
        "round": round_id,
        "selected_clients": selected_clients,
        "selected_malicious": selected_malicious,
        "malicious_kept": malicious_kept,
        "malicious_rejected": malicious_rejected,
        "accuracy": accuracy,
        "loss": loss,
        "relative_change": relative_change,
        "loss_change": loss_change,
        "stable_checks": stable_checks,
        "converged": converged,
    }


# Multi-round experiment

results = []
previous_loss = None
stable_checks = 0

for _ in range(num_rounds):

    result = run_one_round(
        previous_loss=previous_loss,
        stable_checks=stable_checks,
    )

    results.append(result)

    previous_loss = result["loss"]
    stable_checks = result["stable_checks"]

    if result["converged"]:
        print(
            f"\nConverged at round "
            f"{result['round']}"
        )
        break


# Final experiment summary

print("\n" + "=" * 90)
print("EXPERIMENT SUMMARY")
print("=" * 90)

for result in results:

    print(
        f"Round {result['round']:3d} | "
        f"accuracy={result['accuracy']:.4f} | "
        f"loss={result['loss']:.6f} | "
        f"change={result['relative_change']:.6f} | "
        f"stable={result['stable_checks']}/{PATIENCE} | "
        f"malicious selected={len(result['selected_malicious'])} | "
        f"malicious aggregated={len(result['malicious_kept'])}"
    )

print("=" * 90)