from __future__ import annotations

import csv
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ATTACK_TYPE = "sign_flip"
MALICIOUS_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]

DIRICHLET_ALPHA = 0.3

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

RESULTS_DIR = ROOT / "results"
ATTACK_CSV = RESULTS_DIR / "attack_intensity.csv"
PROPOSED_CSV = RESULTS_DIR / "proposed.csv"
PARTITION_CACHE = ROOT / "data" / "partition_cache.pkl"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_START_TIMEOUT = 30


def run_command(args, env, capture_output_metrics=False):

    print("\n>", " ".join(map(str, args)))

    if not capture_output_metrics:
        subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            check=True,
        )
        return {}

    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    metrics = {
        "test_accuracy": None,
        "test_loss": None,
        "macro_f1": None,
        "balanced_accuracy": None,
        "worst_class_accuracy": None,
    }

    patterns = {
        "test_accuracy": re.compile(
            r"Test Accuracy:\s*([0-9]*\.?[0-9]+)"
        ),
        "test_loss": re.compile(
            r"Test Loss:\s*([0-9]*\.?[0-9]+)"
        ),
        "macro_f1": re.compile(
            r"Macro F1:\s*([0-9]*\.?[0-9]+)"
        ),
        "balanced_accuracy": re.compile(
            r"Balanced Accuracy:\s*([0-9]*\.?[0-9]+)"
        ),
        "worst_class_accuracy": re.compile(
            r"Worst-class Accuracy:\s*([0-9]*\.?[0-9]+)"
        ),
    }

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

        for key, pattern in patterns.items():
            match = pattern.search(line)

            if match:
                metrics[key] = float(
                    match.group(1)
                )

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            args,
        )

    return metrics


def wait_for_server(process):
    start = time.time()

    while time.time() - start < SERVER_START_TIMEOUT:

        if process.poll() is not None:
            raise RuntimeError(
                "Server exited before becoming ready."
            )

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as sock:

            sock.settimeout(0.5)

            if (
                sock.connect_ex(
                    (SERVER_HOST, SERVER_PORT)
                )
                == 0
            ):
                print("\nServer is ready.")
                return

        time.sleep(0.5)

    raise TimeoutError(
        "Server did not start within "
        f"{SERVER_START_TIMEOUT} seconds."
    )


def stop_server(process):
    if process is None:
        return

    if process.poll() is not None:
        return

    print("\nStopping server...")

    process.terminate()

    try:
        process.wait(timeout=5)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def write_result(ratio, metrics):
    file_exists = ATTACK_CSV.exists()

    with ATTACK_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "attack_type",
            "malicious_ratio",
            "test_accuracy",
            "test_loss",
            "macro_f1",
            "balanced_accuracy",
            "worst_class_accuracy",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "attack_type":
                    ATTACK_TYPE,

                "malicious_ratio":
                    ratio,

                **metrics,
            }
        )


def plot_results():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = []

    with ATTACK_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                (
                    float(
                        row["malicious_ratio"]
                    ),
                    float(
                        row["test_accuracy"]
                    ),
                )
            )

    rows.sort(
        key=lambda item: item[0]
    )

    attack_percent = [
        ratio * 100
        for ratio, _ in rows
    ]

    accuracies = [
        accuracy * 100
        if accuracy <= 1.0
        else accuracy
        for _, accuracy in rows
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        attack_percent,
        accuracies,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Malicious Client Ratio (%)"
    )

    plt.ylabel(
        "Final Test Accuracy (%)"
    )

    plt.title(
        "Robustness Under Increasing "
        "Attack Intensity"
    )

    plt.xticks(
        attack_percent
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "attack_intensity.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nAttack-intensity plot saved to:"
    )
    print(output_path)


def build_fixed_partition():

    env = os.environ.copy()

    env[
        "DIRICHLET_ALPHA"
    ] = str(
        DIRICHLET_ALPHA
    )

    env[
        "MALICIOUS_RATIO"
    ] = "0"

    env.pop(
        "ATTACK_TYPE",
        None,
    )

    if PARTITION_CACHE.exists():
        PARTITION_CACHE.unlink()

    print(
        "\nBuilding one fixed partition "
        f"with DIRICHLET_ALPHA={DIRICHLET_ALPHA}..."
    )

    run_command(
        [
            PYTHON,
            "-m",
            "data.build_partition_cache",
        ],
        env,
    )


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========================================"
    )
    print(
        "AUTOMATED ATTACK-INTENSITY STUDY"
    )
    print(
        "========================================"
    )

    print(
        "\nAttack type:",
        ATTACK_TYPE,
    )

    print(
        "Malicious ratios:",
        MALICIOUS_RATIOS,
    )

    print(
        "Fixed Dirichlet alpha:",
        DIRICHLET_ALPHA,
    )

    if ATTACK_CSV.exists():
        ATTACK_CSV.unlink()

    build_fixed_partition()

    for index, ratio in enumerate(
        MALICIOUS_RATIOS,
        start=1,
    ):

        print(
            "\n\n########################################"
        )
        print(
            f"RUN {index}/{len(MALICIOUS_RATIOS)}"
        )
        print(
            f"MALICIOUS_RATIO = {ratio}"
        )
        print(
            f"ATTACK_TYPE = {ATTACK_TYPE}"
        )
        print(
            "########################################"
        )

        env = os.environ.copy()

        env[
            "DIRICHLET_ALPHA"
        ] = str(
            DIRICHLET_ALPHA
        )

        env[
            "MALICIOUS_RATIO"
        ] = str(
            ratio
        )


        env[
            "ATTACK_TYPE"
        ] = ATTACK_TYPE

        if PROPOSED_CSV.exists():
            PROPOSED_CSV.unlink()

        server_process = None

        try:
            print(
                "\nStarting fresh server..."
            )

            server_process = (
                subprocess.Popen(
                    [
                        PYTHON,
                        "-m",
                        "server_app.run_server",
                    ],
                    cwd=ROOT,
                    env=env,
                )
            )

            wait_for_server(
                server_process
            )

            run_command(
                [
                    PYTHON,
                    "-m",
                    "client_app.register_all",
                ],
                env,
            )

            metrics = run_command(
                [
                    PYTHON,
                    "main.py",
                ],
                env,
                capture_output_metrics=True,
            )

            if (
                metrics["test_accuracy"]
                is None
            ):
                raise RuntimeError(
                    "Test Accuracy was not found. "
                    "Make sure main.py always runs "
                    "the final test evaluation."
                )

            write_result(
                ratio,
                metrics,
            )

            print(
                f"\nAttack ratio {ratio:.1%} completed."
            )

        finally:
            stop_server(
                server_process
            )

            # Give Windows time to release port 8000.
            time.sleep(2)

    print(
        "\n========================================"
    )
    print(
        "ALL ATTACK-INTENSITY RUNS COMPLETED"
    )
    print(
        "========================================"
    )

    print(
        "\nResults:"
    )
    print(
        ATTACK_CSV
    )

    plot_results()


if __name__ == "__main__":
    main()
