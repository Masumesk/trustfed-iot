from concurrent.futures import ProcessPoolExecutor

import requests

from client_app.api_client import send_update, set_server_url
from client_app.persistent_worker import run_client_round_task
from config import (
    ATTACK_TYPE,
    CLIENT_WORKERS,
    MODEL_CHANGE_THRESHOLD,
    NUM_ROUNDS,
    PATIENCE,
    SERVER_URL,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)
from evaluation.csv_output import save_round_to_csv
#test
from config import EVAL_INTERVAL

SERVER = SERVER_URL
ROUND_CSV_PATH = "results/proposed.csv"


def _header(title, width=72, char="="):
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def _item(label, value):
    print(f"{label:<28} {value}")


def _format_accuracy(value):
    return f"{value:.4f} ({value * 100:.2f}%)"


def _format_trust_summary(trust_scores):

    if not trust_scores:
        return "N/A"

    values = [float(value) for value in trust_scores.values()]
    return (
        f"min={min(values):.4f} | "
        f"avg={sum(values) / len(values):.4f} | "
        f"max={max(values):.4f}"
    )


def _flatten_cluster_clients(cluster_clients):
    client_ids = []
    for clients in cluster_clients.values():
        client_ids.extend(clients)
    return client_ids


def _extract_accepted_ids(accepted_clients):
    accepted_ids = set()

    for cluster_clients in accepted_clients.values():
        for item in cluster_clients:
            client_id = item[0] if isinstance(item, (list, tuple)) else item
            accepted_ids.add(int(client_id))

    return accepted_ids


def train_main_and_backup_clients(
    executor,
    main_ids,
    backup_ids,
    round_id,
):
    main_ids = sorted(set(main_ids))
    backup_ids = sorted(set(backup_ids) - set(main_ids))

    main_futures = {
        client_id: executor.submit(
            run_client_round_task,
            client_id,
            SERVER,
            True,
            round_id,
        )
        for client_id in main_ids
    }

    backup_futures = {
        client_id: executor.submit(
            run_client_round_task,
            client_id,
            SERVER,
            False,
            round_id,
        )
        for client_id in backup_ids
    }

    for client_id, future in main_futures.items():
        result = future.result()

        if result["client_id"] != client_id:
            raise RuntimeError("Client result mismatch.")

        if not result["sent"]:
            raise RuntimeError(f"Main client {client_id} did not send its update.")

    return backup_futures


def send_required_backup_updates(
    backup_requirements,
    backup_futures,
    round_id,
):
    required_ids = {
        client_id
        for client_ids in backup_requirements.values()
        for client_id in client_ids
    }

    sent_ids = []

    for client_id in sorted(required_ids):
        if client_id not in backup_futures:
            raise RuntimeError(
                f"Requested backup {client_id} was not scheduled for training."
            )

        result = backup_futures[client_id].result()

        if result["client_id"] != client_id:
            raise RuntimeError("Backup client result mismatch.")

        if result["sent"]:
            raise RuntimeError(f"Backup client {client_id} sent its update too early.")

        if result["round_id"] != round_id:
            raise RuntimeError(
                "Stale backup update: "
                f"client={client_id}, "
                f"cached_round={result['round_id']}, "
                f"current_round={round_id}"
            )

        send_update(client_id, result["update"], round_id=round_id)
        sent_ids.append(client_id)
        del backup_futures[client_id]

    return sent_ids


def finish_remaining_backups(backup_futures, round_id):
    for client_id, future in list(backup_futures.items()):
        result = future.result()

        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid backup result for client {client_id}")

        if result["client_id"] != client_id:
            raise RuntimeError("Backup client result mismatch.")

        if result["sent"]:
            raise RuntimeError(f"Backup client {client_id} sent its update unexpectedly.")

        if result["round_id"] != round_id:
            raise RuntimeError(
                "Backup round mismatch: "
                f"client={client_id}, "
                f"result_round={result['round_id']}, "
                f"expected_round={round_id}"
            )

    backup_futures.clear()


def _initialize_clustering():
    _header("Proposed | CLUSTERING INITIALIZATION")

    response = requests.post(f"{SERVER}/initialize_clustering")
    response.raise_for_status()
    result = response.json()

    if result.get("status") == "waiting":
        raise RuntimeError("Not enough registered clients for clustering: " f"{result}")

    if result.get("status") not in (
        "clustering initialized",
        "clustering already initialized",
    ):
        raise RuntimeError(f"Clustering initialization failed: {result}")

    clusters = result["clusters"]
    _item("Clusters:", len(clusters))

    for cluster_id, client_ids in clusters.items():
        _item(
            f"Cluster {cluster_id}:",
            f"{len(client_ids)} {client_ids}",
        )


def _start_and_prepare_round(round_id):
    response = requests.post(
        f"{SERVER}/start_round",
        json={"round_id": round_id},
    )
    response.raise_for_status()

    response = requests.post(f"{SERVER}/prepare_round")
    response.raise_for_status()
    prepare_result = response.json()

    if prepare_result.get("status") != "round prepared":
        raise RuntimeError(f"Prepare round failed: {prepare_result}")

    response = requests.get(f"{SERVER}/round_selection")
    response.raise_for_status()
    selection = response.json()

    return {
        "main_ids": _flatten_cluster_clients(selection["main_clients"]),
        "backup_ids": _flatten_cluster_clients(selection["backup_clients"]),
        "representation_fairness": selection["representation_fairness"],
        "hellinger_distance": selection["hellinger_distance"],
    }


def _aggregate_with_backups(backup_futures, round_id):
    print("\n[2/3] Trust evaluation + robust aggregation started...")

    response = requests.post(f"{SERVER}/aggregate")
    response.raise_for_status()
    result = response.json()

    request_cycles = 0
    requested_backup_ids = set()

    while result.get("status") == "backup_needed":
        request_cycles += 1

        if request_cycles > 3:
            raise RuntimeError("Too many backup request cycles.")

        stage = result.get("stage")
        requirements = result["backup_requirements"]
        required_ids = sorted(
            {
                client_id
                for client_ids in requirements.values()
                for client_id in client_ids
            }
        )

        _item(
            f"Backup request ({stage}):",
            f"{len(required_ids)} {required_ids}",
        )

        sent_ids = send_required_backup_updates(
            requirements,
            backup_futures,
            round_id,
        )
        requested_backup_ids.update(sent_ids)

        response = requests.post(f"{SERVER}/aggregate")
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "aggregation completed":
        raise RuntimeError("Aggregation did not complete successfully: " f"{result}")

    print("[2/3] Aggregation completed.")
    return result, requested_backup_ids


def _validate(
    round_id,
    model_relative_change,
    previous_val_loss,
    stable_checks,
):
    val_accuracy = None
    val_loss = None
    val_loss_change = None

    # if model_relative_change < MODEL_CHANGE_THRESHOLD:
    if round_id==1 or round_id % EVAL_INTERVAL == 0 or round_id == 1000:
        print("\n[3/3] Validation started...")

        response = requests.get(f"{SERVER}/evaluate")
        response.raise_for_status()
        evaluation = response.json()

        val_accuracy = float(evaluation["accuracy"])
        val_loss = float(evaluation["loss"])

        if previous_val_loss is not None:
            val_loss_change = abs(val_loss - previous_val_loss)

            if val_loss_change < VAL_LOSS_CHANGE_THRESHOLD:
                stable_checks += 1
            else:
                stable_checks = 0

        previous_val_loss = val_loss
        print("[3/3] Validation completed.")

    else:
        print(
            "\n[3/3] Validation skipped "
            f"(relative change {model_relative_change:.6f} >= "
            f"threshold {MODEL_CHANGE_THRESHOLD:.6f})."
        )
        stable_checks = 0
        previous_val_loss = None

    return (
        val_accuracy,
        val_loss,
        val_loss_change,
        previous_val_loss,
        stable_checks,
        stable_checks >= PATIENCE,
    )


def _print_round_selection(
    main_ids,
    backup_ids,
    selected_malicious,
    representation_fairness,
    hellinger_distance,
):
    _item("Main clients:", f"{len(main_ids)} {main_ids}")
    _item("Backup clients:", f"{len(backup_ids)} {backup_ids}")
    _item(
        "Selected malicious:",
        f"{len(selected_malicious)} {selected_malicious}",
    )
    _item(
        "Representation fairness:",
        f"{representation_fairness * 100:.2f}%",
    )
    _item("Hellinger distance:", f"{hellinger_distance:.4f}")


def _print_round_summary(
    round_id,
    accepted_ids,
    malicious_kept,
    malicious_rejected,
    requested_backup_ids,
    trust_scores,
    model_relative_change,
    val_accuracy,
    val_loss,
    val_loss_change,
    stable_checks,
):
    _header(f"ROUND {round_id} SUMMARY", char="-")

    _item("Accepted clients:", len(accepted_ids))
    _item(
        "Backup updates used:",
        (
            f"{len(requested_backup_ids)} " f"{sorted(requested_backup_ids)}"
            if requested_backup_ids
            else 0
        ),
    )
    _item(
        "Malicious kept:",
        f"{len(malicious_kept)} {malicious_kept}",
    )
    _item(
        "Malicious rejected:",
        f"{len(malicious_rejected)} {malicious_rejected}",
    )
    _item("Trust scores:", _format_trust_summary(trust_scores))
    _item("Relative change:", f"{model_relative_change:.6f}")

    if val_accuracy is None:
        _item("Validation Accuracy:", "SKIPPED")
        _item("Validation Loss:", "SKIPPED")
        _item("Validation loss change:", "N/A")
    else:
        _item("Validation Accuracy:", _format_accuracy(val_accuracy))
        _item("Validation Loss:", f"{val_loss:.6f}")
        _item(
            "Validation loss change:",
            "N/A" if val_loss_change is None else f"{val_loss_change:.6f}",
        )

    _item("Stable checks:", f"{stable_checks}/{PATIENCE}")


def _print_experiment_summary(
    completed_rounds,
    converged,
    last_relative_change,
    last_validation_round,
    last_val_accuracy,
    last_val_loss,
    total_selected_malicious,
    total_malicious_kept,
    total_malicious_rejected,
):
    _header("Proposed | EXPERIMENT SUMMARY")

    _item("Completed rounds:", completed_rounds)
    _item("Stop reason:", "CONVERGED" if converged else "MAX ROUNDS")
    _item("Last relative change:", f"{last_relative_change:.6f}")

    if last_validation_round is None:
        _item("Validation evaluations:", "NONE")
    else:
        _item("Last validation round:", last_validation_round)
        _item(
            "Last validation accuracy:",
            _format_accuracy(last_val_accuracy),
        )
        _item("Last validation loss:", f"{last_val_loss:.6f}")

    _item(
        "Malicious totals:",
        f"selected={total_selected_malicious} | "
        f"kept={total_malicious_kept} | "
        f"rejected={total_malicious_rejected}",
    )


def _final_evaluation():
    _header("Proposed | FINAL TEST EVALUATION")

    response = requests.get(f"{SERVER}/evaluate_final")
    response.raise_for_status()
    evaluation = response.json()

    test_accuracy = float(evaluation["accuracy"])
    test_loss = float(evaluation["loss"])

    _item("Test Accuracy:", _format_accuracy(test_accuracy))
    _item("Test Loss:", f"{test_loss:.6f}")

    macro_f1 = evaluation.get("macro_f1")
    balanced_accuracy = evaluation.get("balanced_accuracy")
    worst_class_accuracy = evaluation.get("worst_class_accuracy")

    if macro_f1 is not None:
        _item("Macro F1:", f"{macro_f1:.4f}")

    if balanced_accuracy is not None:
        _item(
            "Balanced Accuracy:",
            _format_accuracy(float(balanced_accuracy)),
        )

    if worst_class_accuracy is not None:
        _item(
            "Worst-class Accuracy:",
            _format_accuracy(float(worst_class_accuracy)),
        )


def main():
    set_server_url(SERVER)

    if ATTACK_TYPE is None:
        malicious_ids = set()
        print("Attack: NONE")
    else:
        malicious_ids = get_malicious_ids()
        print(f"Attack: {ATTACK_TYPE}")
        print(
            "Malicious clients:",
            sorted(malicious_ids),
        )
    previous_val_loss = None
    stable_checks = 0
    converged = False

    completed_rounds = 0
    last_relative_change = None
    last_validation_round = None
    last_val_accuracy = None
    last_val_loss = None

    total_selected_malicious = 0
    total_malicious_kept = 0
    total_malicious_rejected = 0

    _header("STARTING Proposed EXPERIMENT")
    _item("Server:", SERVER)
    _item("Maximum rounds:", NUM_ROUNDS)
    _item("Client workers:", CLIENT_WORKERS)
    if ATTACK_TYPE is None:
        _item("Attack:", "NONE")
        _item("Configured malicious:", 0)
    else:
        _item("Attack:", ATTACK_TYPE)
        _item(
            "Configured malicious:",
            f"{len(malicious_ids)} "
            f"{sorted(malicious_ids)}",
        )

    _initialize_clustering()

    response = requests.get(f"{SERVER}/evaluate")
    response.raise_for_status()

    evaluation = response.json()

    save_round_to_csv(
        ROUND_CSV_PATH,
        {
            "round": 0,
            "val_accuracy": float(evaluation["accuracy"]),
            "loss": float(evaluation["loss"]),
            "relative_change": 0,
            "representation_fairness": 0,
            "hellinger_distance": 0,
            "selected_malicious": 0,
            "malicious_kept": 0,
            "malicious_rejected": 0,
        },
    )

    with ProcessPoolExecutor(max_workers=CLIENT_WORKERS) as client_executor:
        for round_id in range(1, NUM_ROUNDS + 1):
            _header(f"Proposed | ROUND {round_id}")

            selection = _start_and_prepare_round(round_id)
            main_ids = selection["main_ids"]
            backup_ids = selection["backup_ids"]
            representation_fairness = selection["representation_fairness"]
            hellinger_distance = selection["hellinger_distance"]

            selected_malicious = sorted(set(main_ids + backup_ids) & malicious_ids)

            _print_round_selection(
                main_ids,
                backup_ids,
                selected_malicious,
                representation_fairness,
                hellinger_distance,
            )

            print("\n[1/3] Local training started...")

            backup_futures = train_main_and_backup_clients(
                client_executor,
                main_ids,
                backup_ids,
                round_id,
            )

            print(
                "[1/3] Main updates received; "
                "backup clients remain available on demand."
            )

            aggregation_result, requested_backup_ids = _aggregate_with_backups(
                backup_futures,
                round_id,
            )

            model_relative_change = float(aggregation_result["model_relative_change"])

            (
                val_accuracy,
                val_loss,
                val_loss_change,
                previous_val_loss,
                stable_checks,
                round_converged,
            ) = _validate(
                round_id,
                model_relative_change,
                previous_val_loss,
                stable_checks,
            )

            accepted_ids = _extract_accepted_ids(aggregation_result["accepted_clients"])

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
                ROUND_CSV_PATH,
                {
                    "round": round_id,
                    "val_accuracy": val_accuracy,
                    "loss": val_loss,
                    "relative_change": model_relative_change,
                    "representation_fairness": representation_fairness,
                    "hellinger_distance": hellinger_distance,
                    "selected_malicious": len(selected_malicious),
                    "malicious_kept": len(malicious_kept),
                    "malicious_rejected": len(malicious_rejected),
                },
            )

            finish_remaining_backups(backup_futures, round_id)

            _print_round_summary(
                round_id,
                accepted_ids,
                malicious_kept,
                malicious_rejected,
                requested_backup_ids,
                aggregation_result["trust_scores"],
                model_relative_change,
                val_accuracy,
                val_loss,
                val_loss_change,
                stable_checks,
            )

            completed_rounds = round_id
            last_relative_change = model_relative_change

            total_selected_malicious += len(selected_malicious)
            total_malicious_kept += len(malicious_kept)
            total_malicious_rejected += len(malicious_rejected)

            if val_accuracy is not None:
                last_validation_round = round_id
                last_val_accuracy = val_accuracy
                last_val_loss = val_loss

            # if round_converged:
            #     print(f"\n[Stopping] Converged at round {round_id}.")
            #     converged = True
                # break

    _print_experiment_summary(
        completed_rounds,
        converged,
        last_relative_change,
        last_validation_round,
        last_val_accuracy,
        last_val_loss,
        total_selected_malicious,
        total_malicious_kept,
        total_malicious_rejected,
    )

    _final_evaluation()


if __name__ == "__main__":
    main()
