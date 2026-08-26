import argparse
import json
import subprocess
import sys

import requests

from config import (
    MODEL_CHANGE_THRESHOLD,
    NUM_ROUNDS,
    PATIENCE,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--server",
    type=str,
    default="http://127.0.0.1:8002",
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


# ------------------------------------------------------------------
# Run one complete FL round
# ------------------------------------------------------------------

def run_one_round(
    previous_loss,
    stable_checks,
):

    # 1. Start round
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
    for client_id in selected_clients:
        print(f"\nRunning client {client_id}")

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

    print(
        "\nAll selected clients finished "
        "and sent their updates."
    )

    # 3. Multi-Krum aggregation
    print("\nRunning Multi-Krum aggregation...")

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

    selected_by_krum = aggregation_result.get(
        "selected_by_krum",
        [],
    )

    relative_change = float(
        aggregation_result["relative_change"]
    )

    # 4. Analyze malicious clients (reporting only)
    malicious_kept = [
        client_id
        for client_id in selected_by_krum
        if client_id in malicious_ids
    ]

    malicious_rejected = [
        client_id
        for client_id in selected_malicious
        if client_id not in selected_by_krum
    ]

    # 5. Evaluate new global model
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

    # 6. Convergence check -- same logic as standalone baseline
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

    # 7. Round summary
    print("\n" + "-" * 70)
    print(f"ROUND {round_id} SUMMARY")
    print("-" * 70)
    print(f"Selected clients:       {selected_clients}")
    print(f"Selected malicious:     {selected_malicious}")
    print(f"Selected by Multi-Krum: {selected_by_krum}")
    print(f"Malicious kept:         {malicious_kept}")
    print(f"Malicious rejected:     {malicious_rejected}")
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
        "selected_by_krum": selected_by_krum,
        "malicious_kept": malicious_kept,
        "malicious_rejected": malicious_rejected,
        "accuracy": accuracy,
        "loss": loss,
        "relative_change": relative_change,
        "loss_change": loss_change,
        "stable_checks": stable_checks,
        "converged": converged,
    }


# ------------------------------------------------------------------
# Multi-round experiment
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Final experiment summary
# ------------------------------------------------------------------

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
        f"malicious kept={len(result['malicious_kept'])}"
    )

print("=" * 90)
