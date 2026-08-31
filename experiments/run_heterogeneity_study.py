from __future__ import annotations

import csv
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ALPHAS = [0.1, 0.3, 0.5, 1.0]

ROOT = Path(__file__).resolve().parent.parent

PYTHON = sys.executable

RESULTS_DIR = ROOT / "results"
HETEROGENEITY_CSV = RESULTS_DIR / "heterogeneity.csv"
PROPOSED_CSV = RESULTS_DIR / "proposed.csv"
PARTITION_CACHE = ROOT / "data" / "partition_cache.pkl"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_START_TIMEOUT = 30


def run_command(args, env, capture_test_accuracy=False):

    print("\n>", " ".join(map(str, args)))

    if not capture_test_accuracy:
        subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            check=True,
        )
        return None

    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    test_accuracy = None

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

        match = re.search(
            r"Test Accuracy:\s*([0-9]*\.?[0-9]+)",
            line,
        )

        if match:
            test_accuracy = float(match.group(1))

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            args,
        )

    return test_accuracy


def wait_for_server(process):
    """Wait until port 8000 is ready."""

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
    """Stop the server process safely."""

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


def csv_data_row_count(path):
    """Return number of data rows in a CSV file."""

    if not path.exists():
        return 0

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)

    if not rows:
        return 0

    return max(0, len(rows) - 1)


def append_fallback_result(alpha, test_accuracy):
    """
    If main.py did not write heterogeneity.csv,
    save alpha + Test Accuracy here.
    """

    file_exists = HETEROGENEITY_CSV.exists()

    with HETEROGENEITY_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dirichlet_alpha",
                "test_accuracy",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "dirichlet_alpha": alpha,
                "test_accuracy": test_accuracy,
            }
        )


def plot_results():
    """Create the final heterogeneity plot."""

    if not HETEROGENEITY_CSV.exists():
        raise FileNotFoundError(
            f"Result file not found: "
            f"{HETEROGENEITY_CSV}"
        )

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = []

    with HETEROGENEITY_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            alpha = float(
                row["dirichlet_alpha"]
            )

            accuracy = float(
                row["test_accuracy"]
            )

            rows.append(
                (alpha, accuracy)
            )

    if not rows:
        raise RuntimeError(
            "heterogeneity.csv has no data rows."
        )

    rows.sort(
        key=lambda item: item[0]
    )

    alphas = [
        item[0]
        for item in rows
    ]

    accuracies = [
        item[1] * 100
        if item[1] <= 1.0
        else item[1]
        for item in rows
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        alphas,
        accuracies,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Dirichlet Alpha"
    )

    plt.ylabel(
        "Final Test Accuracy (%)"
    )

    plt.title(
        "Effect of Data Heterogeneity "
        "on Model Performance"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "heterogeneity.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nHeterogeneity plot saved to:"
    )
    print(output_path)


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========================================"
    )
    print(
        "AUTOMATED DATA HETEROGENEITY STUDY"
    )
    print(
        "========================================"
    )

    print(
        "\nDirichlet alpha values:",
        ALPHAS,
    )

    # Start with a clean heterogeneity result file.
    if HETEROGENEITY_CSV.exists():
        HETEROGENEITY_CSV.unlink()

    for index, alpha in enumerate(
        ALPHAS,
        start=1,
    ):

        print(
            "\n\n########################################"
        )
        print(
            f"RUN {index}/{len(ALPHAS)}"
        )
        print(
            f"DIRICHLET_ALPHA = {alpha}"
        )
        print(
            "########################################"
        )

        env = os.environ.copy()

        # Fix attack settings so this experiment
        # measures only data heterogeneity.
        env[
            "DIRICHLET_ALPHA"
        ] = str(alpha)

        env[
            "MALICIOUS_RATIO"
        ] = "0"

        env.pop(
            "ATTACK_TYPE",
            None,
        )

        # Force a fresh partition for this alpha.
        if PARTITION_CACHE.exists():

            print(
                "\nDeleting old partition cache:"
            )
            print(
                PARTITION_CACHE
            )

            PARTITION_CACHE.unlink()

        # Keep round-level results separate
        # between alpha runs.
        if PROPOSED_CSV.exists():
            PROPOSED_CSV.unlink()

        run_command(
            [
                PYTHON,
                "-m",
                "data.build_partition_cache",
            ],
            env,
        )

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

            # Register all clients against
            # the fresh server.
            run_command(
                [
                    PYTHON,
                    "-m",
                    "client_app.register_all",
                ],
                env,
            )

            rows_before = (
                csv_data_row_count(
                    HETEROGENEITY_CSV
                )
            )

            # Run the complete FL training.
            test_accuracy = run_command(
                [
                    PYTHON,
                    "main.py",
                ],
                env,
                capture_test_accuracy=True,
            )

            rows_after = (
                csv_data_row_count(
                    HETEROGENEITY_CSV
                )
            )

            # If main.py already contains the
            # heterogeneity CSV saving block,
            # it has already written the result.
            #
            # Otherwise, fall back to the Test
            # Accuracy printed by main.py.
            if rows_after == rows_before:

                if test_accuracy is None:
                    raise RuntimeError(
                        "No Test Accuracy was found "
                        "and main.py did not write "
                        "heterogeneity.csv."
                    )

                append_fallback_result(
                    alpha,
                    test_accuracy,
                )

            print(
                f"\nAlpha {alpha} completed."
            )

        finally:
            stop_server(
                server_process
            )

            # Small pause so Windows fully releases
            # port 8000 before the next run.
            time.sleep(2)

    print(
        "\n========================================"
    )
    print(
        "ALL HETEROGENEITY RUNS COMPLETED"
    )
    print(
        "========================================"
    )

    print(
        "\nResults:"
    )
    print(
        HETEROGENEITY_CSV
    )

    plot_results()


if __name__ == "__main__":
    main()
