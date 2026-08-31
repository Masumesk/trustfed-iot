import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import requests

from client_app.persistent_worker import (
    run_client_round_task,
)

from config import (
    CLIENT_WORKERS,
    MODEL_CHANGE_THRESHOLD,
    NUM_ROUNDS,
    PATIENCE,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)

from evaluation.csv_output import (
    save_round_to_csv,
)



# Run one Multi-Krum round


def run_one_round(
    server_url,
    malicious_ids,
    previous_val_loss,
    stable_checks,
    client_executor,
):

    
    # Start round
    

    response = requests.post(
        f"{server_url}/start_round"
    )

    response.raise_for_status()

    round_info = response.json()

    round_id = round_info["round"]

    selected_clients = round_info[
        "selected_clients"
    ]


    selected_malicious = [
        client_id
        for client_id in selected_clients
        if client_id in malicious_ids
    ]


    print("\n" + "=" * 70)
    print(f"ROUND {round_id}")
    print("=" * 70)

    print(
        f"Selected clients:   "
        f"{selected_clients}"
    )

    print(
        f"Selected malicious: "
        f"{selected_malicious}"
    )


    
    # Persistent client workers
    

    futures = [
        client_executor.submit(
            run_client_round_task,
            client_id,
            server_url,
        )
        for client_id in selected_clients
    ]


    for future in futures:
        future.result()


    print(
        "\nAll selected clients finished "
        "and sent their updates."
    )


    
    # Multi-Krum aggregation
    

    print(
        "\nRunning Multi-Krum aggregation..."
    )

    response = requests.post(
        f"{server_url}/aggregate"
    )

    response.raise_for_status()

    aggregation_result = (
        response.json()
    )


    print("\nAggregation result:")

    print(
        json.dumps(
            aggregation_result,
            indent=2,
        )
    )


    selected_by_krum = (
        aggregation_result.get(
            "selected_by_krum",
            [],
        )
    )


    relative_change = float(
        aggregation_result[
            "relative_change"
        ]
    )


    
    # Multi-Krum malicious statistics
    

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


    
    # Validation
    

    val_accuracy = None
    val_loss = None
    val_loss_change = None


    if (
        relative_change
        <
        MODEL_CHANGE_THRESHOLD
    ):

        print(
            "\nModel change is below threshold."
        )

        print(
            "Evaluating global model "
            "on VALIDATION set..."
        )


        response = requests.get(
            f"{server_url}/evaluate"
        )

        response.raise_for_status()

        evaluation = response.json()


        val_accuracy = float(
            evaluation["accuracy"]
        )

        val_loss = float(
            evaluation["loss"]
        )


        if previous_val_loss is not None:

            val_loss_change = abs(
                val_loss
                -
                previous_val_loss
            )


            if (
                val_loss_change
                <
                VAL_LOSS_CHANGE_THRESHOLD
            ):

                stable_checks += 1

            else:

                stable_checks = 0


        previous_val_loss = val_loss


    else:

        print(
            "\nModel change is above threshold."
        )

        print(
            "Skipping validation evaluation."
        )

        stable_checks = 0
        previous_val_loss = None


    converged = (
        stable_checks >= PATIENCE
    )


    
    # Round summary
    

    print("\n" + "-" * 70)
    print(f"ROUND {round_id} SUMMARY")
    print("-" * 70)


    print(
        f"Selected clients:       "
        f"{selected_clients}"
    )

    print(
        f"Selected malicious:     "
        f"{selected_malicious}"
    )

    print(
        f"Selected by Multi-Krum: "
        f"{selected_by_krum}"
    )

    print(
        f"Malicious kept:         "
        f"{malicious_kept}"
    )

    print(
        f"Malicious rejected:     "
        f"{malicious_rejected}"
    )

    print(
        f"Relative change:        "
        f"{relative_change:.6f}"
    )


    if val_accuracy is None:

        print(
            "Validation Accuracy:    SKIPPED"
        )

        print(
            "Validation Loss:        SKIPPED"
        )

        print(
            "Validation loss change: N/A"
        )


    else:

        print(
            f"Validation Accuracy:    "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation Loss:        "
            f"{val_loss:.6f}"
        )


        if val_loss_change is None:

            print(
                "Validation loss change: N/A"
            )

        else:

            print(
                f"Validation loss change: "
                f"{val_loss_change:.6f}"
            )


    print(
        f"Stable:                 "
        f"{stable_checks}/{PATIENCE}"
    )

    print("-" * 70)


    
    # Save round result
    

    save_round_to_csv(
        "results/multikrum.csv",
        {
            "round":
                round_id,

            "val_accuracy":
                val_accuracy,

            "val_loss":
                val_loss,

            "relative_change":
                relative_change,

            "selected_malicious":
                len(
                    selected_malicious
                ),

            "malicious_kept":
                len(
                    malicious_kept
                ),

            "malicious_rejected":
                len(
                    malicious_rejected
                ),
        }
    )


    return {
        "round":
            round_id,

        "selected_clients":
            selected_clients,

        "selected_malicious":
            selected_malicious,

        "selected_by_krum":
            selected_by_krum,

        "malicious_kept":
            malicious_kept,

        "malicious_rejected":
            malicious_rejected,

        "val_accuracy":
            val_accuracy,

        "val_loss":
            val_loss,

        "relative_change":
            relative_change,

        "val_loss_change":
            val_loss_change,

        "previous_val_loss":
            previous_val_loss,

        "stable_checks":
            stable_checks,

        "converged":
            converged,
    }



# Main


def main():

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


    server_url = (
        args.server.rstrip("/")
    )

    num_rounds = (
        args.rounds
    )


    malicious_ids = set(
        get_malicious_ids()
    )


    results = []

    previous_val_loss = None

    stable_checks = 0

    converged = False


    
    # Persistent ProcessPool
    # Created once for all rounds
    

    with ProcessPoolExecutor(
        max_workers=CLIENT_WORKERS
    ) as client_executor:


        for _ in range(
            num_rounds
        ):

            result = run_one_round(
                server_url=
                    server_url,

                malicious_ids=
                    malicious_ids,

                previous_val_loss=
                    previous_val_loss,

                stable_checks=
                    stable_checks,

                client_executor=
                    client_executor,
            )


            results.append(
                result
            )


            previous_val_loss = (
                result[
                    "previous_val_loss"
                ]
            )


            stable_checks = (
                result[
                    "stable_checks"
                ]
            )


            if result["converged"]:

                converged = True

                print(
                    f"\nConverged at round "
                    f"{result['round']}"
                )

                break


    
    # Experiment summary
    

    print("\n" + "=" * 90)

    print(
        "EXPERIMENT SUMMARY"
    )

    print("=" * 90)


    for result in results:

        if (
            result["val_accuracy"]
            is None
        ):

            val_accuracy_text = (
                "SKIPPED"
            )

            val_loss_text = (
                "SKIPPED"
            )


        else:

            val_accuracy_text = (
                f"{result['val_accuracy']:.4f}"
            )

            val_loss_text = (
                f"{result['val_loss']:.6f}"
            )


        print(
            f"Round "
            f"{result['round']:3d} | "
            f"val_accuracy="
            f"{val_accuracy_text} | "
            f"val_loss="
            f"{val_loss_text} | "
            f"change="
            f"{result['relative_change']:.6f} | "
            f"stable="
            f"{result['stable_checks']}/{PATIENCE} | "
            f"malicious selected="
            f"{len(result['selected_malicious'])} | "
            f"malicious kept="
            f"{len(result['malicious_kept'])}"
        )


    print("=" * 90)


    
    # Final TEST evaluation
    



    print("\n" + "=" * 70)

    print(
        "FINAL EVALUATION ON TEST SET"
    )

    print("=" * 70)


    response = requests.get(
        f"{server_url}/evaluate_final"
    )

    response.raise_for_status()

    test_result = (
        response.json()
    )


    test_accuracy = float(
        test_result[
            "accuracy"
        ]
    )


    test_loss = float(
        test_result[
            "loss"
        ]
    )


    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.6f}"
    )


    if (
        "macro_f1"
        in test_result
    ):

        print(
            f"Macro F1: "
            f"{test_result['macro_f1']:.4f}"
        )


    if (
        "balanced_accuracy"
        in test_result
    ):

        print(
            f"Balanced Accuracy: "
            f"{test_result['balanced_accuracy']:.4f}"
        )


    if (
        "worst_class_accuracy"
        in test_result
    ):

        print(
            f"Worst-class Accuracy: "
            f"{test_result['worst_class_accuracy']:.4f}"
        )


    save_round_to_csv(
        "results/multikrum_final_test.csv",
        {
            "round":
                results[-1]["round"],

            "test_accuracy":
                test_accuracy,

            "test_loss":
                test_loss,

            "macro_f1":
                test_result.get(
                    "macro_f1"
                ),

            "balanced_accuracy":
                test_result.get(
                    "balanced_accuracy"
                ),

            "worst_class_accuracy":
                test_result.get(
                    "worst_class_accuracy"
                ),
        }
    )







if __name__ == "__main__":
    main()