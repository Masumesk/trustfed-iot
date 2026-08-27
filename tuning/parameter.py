import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


ALLOWED_PARAMETERS = {
    "LAMBDA_TRUST",
    "TRUST_THRESHOLD",
    "SELECTION_ALPHA",
    "T_NEAR",
    "TRIM_RATIO",
    "OPTICS_XI",
    "OPTICS_MIN_SAMPLES",
    "NOISE_ASSIGNMENT_THRESHOLD",
    "RANDOM_RATIO",
    "PARTICIPATION_RATIO",
}


DEFAULT_VALUES = {
    "LAMBDA_TRUST": [0.2, 0.4, 0.5, 0.6, 0.8],
    "TRUST_THRESHOLD": [0.3, 0.4, 0.5, 0.6, 0.7],
    "SELECTION_ALPHA": [0.2, 0.4, 0.5, 0.6, 0.8],
    "T_NEAR": [0.5, 0.6, 0.7, 0.8, 0.9],
    "TRIM_RATIO": [0.0, 0.1, 0.2, 0.3],
    "OPTICS_XI": [0.03, 0.05, 0.07, 0.10],
    "OPTICS_MIN_SAMPLES": [2, 3, 4, 5],
    "NOISE_ASSIGNMENT_THRESHOLD": [0.5, 0.6, 0.7, 0.8],
    "RANDOM_RATIO": [0.25, 0.5, 0.75],
    "PARTICIPATION_RATIO": [0.2, 0.3, 0.4, 0.5],
}


from config import SERVER_URL as SERVER


def wait_for_server(
    server_process,
    timeout=60
):
    start_time = time.time()

    while time.time() - start_time < timeout:

        if server_process.poll() is not None:
            raise RuntimeError(
                "Server stopped before becoming ready."
            )

        try:
            response = requests.get(
                SERVER,
                timeout=1
            )

            if response.ok:
                return

        except requests.RequestException:
            pass

        time.sleep(0.5)

    raise TimeoutError(
        "Server did not become ready within "
        f"{timeout} seconds."
    )


def stop_process(process):

    if process is None:
        return

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=10)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def read_csv_rows(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        return list(
            csv.DictReader(file)
        )


def safe_float(value):

    if value in (
        None,
        "",
        "None"
    ):
        return None

    return float(value)


def summarize_run(result_csv, parameter, parameter_value):
    rows = read_csv_rows(result_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {result_csv}")

    final_row = rows[-1]
    accuracies = [float(r["accuracy"]) for r in rows]
    n = min(5, len(accuracies))

    # Malicious stats
    def _sum_int(key):
        return sum(int(float(r.get(key, 0) or 0)) for r in rows)

    sel_mal = _sum_int("selected_malicious")
    rej_mal = _sum_int("malicious_rejected")
    kept_mal = _sum_int("malicious_kept")
    rej_rate = rej_mal / sel_mal if sel_mal > 0 else 0.0
    acc_rate = kept_mal / sel_mal if sel_mal > 0 else 0.0

    # Convergence round
    conv_round = ""
    for r in rows:
        if str(r.get("converged", "0")) in ("1", "1.0", "True", "true"):
            conv_round = int(float(r["round"]))
            break

    # Mean evaluation time
    eval_times = [v for v in (safe_float(r.get("evaluation_time_sec")) for r in rows) if v is not None]
    mean_eval = sum(eval_times) / len(eval_times) if eval_times else None

    # Optional float fields from final row
    FLOAT_FIELDS = [
        "balanced_accuracy", "worst_class_accuracy", "class_accuracy_std",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "relative_change",
    ]
    float_vals = {f"final_{k}": safe_float(final_row.get(k)) for k in FLOAT_FIELDS}

    return {
        "parameter": parameter,
        "value": parameter_value,
        "rounds_completed": int(float(final_row["round"])),
        "convergence_round": conv_round,
        "final_accuracy": float(final_row["accuracy"]),
        "mean_last_5_accuracy": sum(accuracies[-n:]) / n,
        "best_accuracy": max(accuracies),
        "final_loss": float(final_row["loss"]),
        **float_vals,
        "malicious_rejection_rate": rej_rate,
        "malicious_acceptance_rate": acc_rate,
        "selected_malicious_total": sel_mal,
        "malicious_rejected_total": rej_mal,
        "malicious_kept_total": kept_mal,
        "mean_evaluation_time_sec": mean_eval,
    }


def save_summary(
    summary_rows,
    output_path
):

    if not summary_rows:
        return

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )


def run_single_experiment(project_dir, parameter, parameter_value, run_dir):
    env = os.environ.copy()
    env[parameter] = str(parameter_value)

    result_file = project_dir / "results" / "proposed.csv"
    if result_file.exists():
        result_file.unlink()

    server_log = run_dir / "server.log"
    register_log = run_dir / "register.log"
    main_log = run_dir / "main.log"
    server_process = None

    try:
        print(f"Running {parameter}={parameter_value}")

        with open(server_log, "w", encoding="utf-8") as sl:
            server_process = subprocess.Popen(
                [sys.executable, "-m", "server_app.run_server"],
                cwd=project_dir, env=env, stdout=sl, stderr=subprocess.STDOUT,
            )
            wait_for_server(server_process)

        with open(register_log, "w", encoding="utf-8") as rl:
            subprocess.run(
                [sys.executable, "-m", "client_app.register_all"],
                cwd=project_dir, env=env, stdout=rl, stderr=subprocess.STDOUT, check=True,
            )

        with open(main_log, "w", encoding="utf-8") as ml:
            subprocess.run(
                [sys.executable, "main.py"],
                cwd=project_dir, env=env, stdout=ml, stderr=subprocess.STDOUT, check=True,
            )

        if not result_file.exists():
            raise RuntimeError("results/proposed.csv not created")

        copied = run_dir / "proposed.csv"
        shutil.copy2(result_file, copied)
        return copied

    finally:
        stop_process(server_process)
