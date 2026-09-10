from concurrent.futures import ProcessPoolExecutor

import requests
import math

from client_app.persistent_worker import run_client_round_task
from config import (
    ATTACK_TYPE,
    CLIENT_WORKERS,
    MODEL_CHANGE_THRESHOLD,
    PATIENCE,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)
from evaluation.csv_output import save_round_to_csv
from config import EVAL_INTERVAL

FEDAVG = "fedavg"
MULTI_KRUM = "multikrum"

SUPPORTED_METHODS = {
    FEDAVG,
    MULTI_KRUM,
}


def _header(title, width=72, char="="):
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def _item(label, value):
    print(f"{label:<27} {value}")


def _format_accuracy(value):
    return f"{value:.4f} ({value * 100:.2f}%)"

def _safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _start_round(server_url, malicious_ids, method_name):
    response = requests.post(f"{server_url}/start_round")
    response.raise_for_status()
    round_info = response.json()

    round_id = round_info["round"]
    selected_clients = round_info["selected_clients"]
    selected_malicious = [
        client_id
        for client_id in selected_clients
        if client_id in malicious_ids
    ]

    _header(f"{method_name} | ROUND {round_id}")
    _item(
        "Selected clients:",
        f"{len(selected_clients)} {selected_clients}",
    )
    _item(
        "Selected malicious:",
        f"{len(selected_malicious)} {selected_malicious}",
    )

    return round_id, selected_clients, selected_malicious


def _run_clients(
    server_url,
    round_id,
    selected_clients,
    client_executor,
):
    print("\n[1/3] Local training started...")

    futures = [
        client_executor.submit(
            run_client_round_task,
            client_id,
            server_url,
            True,
            round_id,
        )
        for client_id in selected_clients
    ]

    for future in futures:
        future.result()

    print("[1/3] Local training completed.")


def _aggregate(
    server_url,
    method,
    method_name,
    malicious_ids,
    selected_malicious,
):
    print(f"\n[2/3] {method_name} aggregation started...")

    response = requests.post(f"{server_url}/aggregate")
    response.raise_for_status()
    aggregation_result = response.json()

    relative_change = float(aggregation_result["relative_change"])

    if method == FEDAVG:
        selected_by_krum = None
        malicious_kept = list(selected_malicious)
        malicious_rejected = []

    elif method == MULTI_KRUM:
        selected_by_krum = aggregation_result.get(
            "selected_by_krum",
            [],
        )
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

    else:
        raise ValueError(f"Unsupported method: {method}")

    print("[2/3] Aggregation completed.")

    return (
        relative_change,
        selected_by_krum,
        malicious_kept,
        malicious_rejected,
    )


def _validate(
    round_id,
    server_url,
    relative_change,
    previous_val_loss,
    stable_checks,
):
    val_accuracy = None
    val_loss = None
    val_loss_change = None

    if round_id == 1 or round_id % EVAL_INTERVAL == 0 or round_id == 1000:
        print("\n[3/3] Validation started...")

        try:
            response = requests.get(
                f"{server_url}/evaluate",
                timeout=300,
            )

            response.raise_for_status()
            evaluation = response.json()

            val_accuracy = _safe_float(
                evaluation.get("accuracy")
            )

            val_loss = _safe_float(
                evaluation.get("loss")
            )

            status = evaluation.get(
                "status",
                "ok",
            )

            if status != "ok":
                print(
                    f"[WARNING] Validation status: {status}"
                )
                
            if val_loss is not None:

                if previous_val_loss is not None:
                    val_loss_change = abs(
                        val_loss - previous_val_loss
                    )

                    if (
                        val_loss_change
                        < VAL_LOSS_CHANGE_THRESHOLD
                    ):
                        stable_checks += 1
                    else:
                        stable_checks = 0

                previous_val_loss = val_loss

            else:
                stable_checks = 0
                previous_val_loss = None

            print("[3/3] Validation completed.")

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as exc:

            print(
                "[WARNING] Validation failed, "
                "but training will continue."
            )
            print(
                f"[WARNING] Reason: {exc}"
            )

            val_accuracy = None
            val_loss = None
            val_loss_change = None
            stable_checks = 0
            previous_val_loss = None

    else:
        print("\n[3/3] Validation skipped.")

        stable_checks = 0
        previous_val_loss = None

    converged = stable_checks >= PATIENCE

    return (
        val_accuracy,
        val_loss,
        val_loss_change,
        previous_val_loss,
        stable_checks,
        converged,
    )

def _print_round_summary(
    method,
    round_id,
    selected_clients,
    selected_malicious,
    selected_by_krum,
    malicious_kept,
    malicious_rejected,
    relative_change,
    val_accuracy,
    val_loss,
    val_loss_change,
    stable_checks,
):
    _header(
        f"ROUND {round_id} SUMMARY",
        char="-",
    )

    _item("Selected clients:", len(selected_clients))
    _item("Selected malicious:", len(selected_malicious))

    if method == MULTI_KRUM:
        _item(
            "Selected by Multi-Krum:",
            f"{len(selected_by_krum)} {selected_by_krum}",
        )

    _item(
        "Malicious kept:",
        f"{len(malicious_kept)} {malicious_kept}",
    )
    _item(
        "Malicious rejected:",
        f"{len(malicious_rejected)} {malicious_rejected}",
    )
    _item(
        "Relative change:",
        f"{relative_change:.6f}",
    )

    if val_accuracy is None:
        _item(
            "Validation Accuracy:",
            "N/A",
        )
    else:
        _item(
            "Validation Accuracy:",
            _format_accuracy(val_accuracy),
        )

    if val_loss is None:
        _item(
            "Validation Loss:",
            "N/A",
        )
    else:
        _item(
            "Validation Loss:",
            f"{val_loss:.6f}",
        )

    _item(
        "Validation loss change:",
        (
            "N/A"
            if val_loss_change is None
            else f"{val_loss_change:.6f}"
        ),
    )
    _item(
        "Stable checks:",
        f"{stable_checks}/{PATIENCE}",
    )


def run_one_round(
    *,
    server_url,
    method,
    method_name,
    round_csv_path,
    malicious_ids,
    previous_val_loss,
    stable_checks,
    client_executor,
):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    (
        round_id,
        selected_clients,
        selected_malicious,
    ) = _start_round(
        server_url,
        malicious_ids,
        method_name,
    )

    _run_clients(
        server_url,
        round_id,
        selected_clients,
        client_executor,
    )

    (
        relative_change,
        selected_by_krum,
        malicious_kept,
        malicious_rejected,
    ) = _aggregate(
        server_url,
        method,
        method_name,
        malicious_ids,
        selected_malicious,
    )

    (
        val_accuracy,
        val_loss,
        val_loss_change,
        previous_val_loss,
        stable_checks,
        converged,
    ) = _validate(
        round_id,
        server_url,
        relative_change,
        previous_val_loss,
        stable_checks,
    )

    _print_round_summary(
        method,
        round_id,
        selected_clients,
        selected_malicious,
        selected_by_krum,
        malicious_kept,
        malicious_rejected,
        relative_change,
        val_accuracy,
        val_loss,
        val_loss_change,
        stable_checks,
    )

    save_round_to_csv(
        round_csv_path,
        {
            "round": round_id,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "relative_change": relative_change,
            "selected_malicious": len(selected_malicious),
            "malicious_kept": len(malicious_kept),
            "malicious_rejected": len(malicious_rejected),
        },
    )

    result = {
        "round": round_id,
        "selected_clients": selected_clients,
        "selected_malicious": selected_malicious,
        "malicious_kept": malicious_kept,
        "malicious_rejected": malicious_rejected,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "relative_change": relative_change,
        "val_loss_change": val_loss_change,
        "previous_val_loss": previous_val_loss,
        "stable_checks": stable_checks,
        "converged": converged,
    }

    if method == MULTI_KRUM:
        result["selected_by_krum"] = selected_by_krum

    return result


def _print_experiment_summary(method_name, results):
    _header(
        f"{method_name} | EXPERIMENT SUMMARY",
        width=72,
    )

    completed_rounds = len(results)
    last_result = results[-1]
    evaluated_results = [
        result
        for result in results
        if result["val_accuracy"] is not None
    ]

    total_selected_malicious = sum(
        len(result["selected_malicious"])
        for result in results
    )
    total_kept = sum(
        len(result["malicious_kept"])
        for result in results
    )
    total_rejected = sum(
        len(result["malicious_rejected"])
        for result in results
    )

    _item("Completed rounds:", completed_rounds)
    _item(
        "Stop reason:",
        "CONVERGED" if last_result["converged"] else "MAX ROUNDS",
    )
    _item(
        "Last relative change:",
        f"{last_result['relative_change']:.6f}",
    )

    if evaluated_results:
        last_evaluated = evaluated_results[-1]
        _item(
            "Last validation round:",
            last_evaluated["round"],
        )
        _item(
            "Last validation accuracy:",
            _format_accuracy(last_evaluated["val_accuracy"]),
        )
        last_val_loss = last_evaluated["val_loss"]
        _item(
            "Last validation loss:",
            (
                "N/A"
                if last_val_loss is None
                else f"{last_val_loss:.6f}"
            ),
        )
    else:
        _item("Validation evaluations:", "NONE")

    _item(
        "Malicious totals:",
        f"selected={total_selected_malicious} | "
        f"kept={total_kept} | "
        f"rejected={total_rejected}",
    )


def _final_evaluation(
    server_url,
    method_name,
    final_csv_path,
    last_round,
):
    _header(f"{method_name} | FINAL TEST EVALUATION")

    test_accuracy = None
    test_loss = None
    macro_f1 = None
    balanced_accuracy = None
    worst_class_accuracy = None

    try:
        response = requests.get(
            f"{server_url}/evaluate_final",
            timeout=300,
        )

        response.raise_for_status()
        test_result = response.json()

        test_accuracy = _safe_float(
            test_result.get("accuracy")
        )

        test_loss = _safe_float(
            test_result.get("loss")
        )

        macro_f1 = _safe_float(
            test_result.get("macro_f1")
        )

        balanced_accuracy = _safe_float(
            test_result.get("balanced_accuracy")
        )

        worst_class_accuracy = _safe_float(
            test_result.get("worst_class_accuracy")
        )

        status = test_result.get(
            "status",
            "ok",
        )

        if status != "ok":
            print(
                f"[WARNING] Final evaluation status: {status}"
            )

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:

        print(
            "[WARNING] Final evaluation failed, "
            "but experiment results will still be saved."
        )

        print(
            f"[WARNING] Reason: {exc}"
        )

    _item(
        "Test Accuracy:",
        (
            "N/A"
            if test_accuracy is None
            else _format_accuracy(test_accuracy)
        ),
    )

    _item(
        "Test Loss:",
        (
            "N/A"
            if test_loss is None
            else f"{test_loss:.6f}"
        ),
    )

    _item(
        "Macro F1:",
        (
            "N/A"
            if macro_f1 is None
            else f"{macro_f1:.4f}"
        ),
    )

    _item(
        "Balanced Accuracy:",
        (
            "N/A"
            if balanced_accuracy is None
            else _format_accuracy(balanced_accuracy)
        ),
    )

    _item(
        "Worst-class Accuracy:",
        (
            "N/A"
            if worst_class_accuracy is None
            else _format_accuracy(worst_class_accuracy)
        ),
    )

    save_round_to_csv(
        final_csv_path,
        {
            "round": last_round,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "worst_class_accuracy": worst_class_accuracy,
        },
    )

def run_experiment(
    *,
    server_url,
    num_rounds,
    method,
    method_name,
    round_csv_path,
    final_csv_path,
):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    server_url = server_url.rstrip("/")
    if ATTACK_TYPE is None:
        malicious_ids = set()
    else:
        malicious_ids = set(get_malicious_ids())

    results = []
    previous_val_loss = None
    stable_checks = 0

    # Evaluate initial model (round 0)
    response = requests.get(f"{server_url}/evaluate")
    response.raise_for_status()

    evaluation = response.json()

    save_round_to_csv(
        round_csv_path,
        {
            "round": 0,
            "val_accuracy": float(evaluation["accuracy"]),
            "val_loss": float(evaluation["loss"]),
            "relative_change": 0,
            "selected_malicious": 0,
            "malicious_kept": 0,
            "malicious_rejected": 0,
        },
    )

    _header(f"STARTING {method_name} EXPERIMENT")
    _item("Server:", server_url)
    _item("Maximum rounds:", num_rounds)
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

    with ProcessPoolExecutor(max_workers=CLIENT_WORKERS) as client_executor:
        for _ in range(num_rounds):
            result = run_one_round(
                server_url=server_url,
                method=method,
                method_name=method_name,
                round_csv_path=round_csv_path,
                malicious_ids=malicious_ids,
                previous_val_loss=previous_val_loss,
                stable_checks=stable_checks,
                client_executor=client_executor,
            )

            results.append(result)
            previous_val_loss = result["previous_val_loss"]
            stable_checks = result["stable_checks"]

            # if result["converged"]:
            #     print(
            #         f"\n[Stopping] Converged at round {result['round']}."
            #     )
            #     break

    _print_experiment_summary(
        method_name,
        results,
    )

    _final_evaluation(
        server_url,
        method_name,
        final_csv_path,
        results[-1]["round"],
    )

    return results
