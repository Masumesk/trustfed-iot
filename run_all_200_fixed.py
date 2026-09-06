#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_all_200.py
==============
Runs TrustFed-IoT comparison experiments sequentially:
    1) FedAvg
    2) Multi-Krum
    3) Proposed method (main.py)

Then parses the logs and creates:
    - final_metrics.csv
    - round_metrics.csv
    - final_metrics_comparison.png
    - relative_change_comparison.png
    - periodic_accuracy_comparison.png (when enough points exist)
    - summary.txt
    - run_manifest.json

Designed for the repository root:
    D:\workspace\Thesis\trustfed-iot

Important:
- Start all three FastAPI servers BEFORE this script:
    Proposed   -> http://127.0.0.1:8000
    FedAvg     -> http://127.0.0.1:8001
    Multi-Krum -> http://127.0.0.1:8002
- Run this script from the active .venv-gpu environment.
- Experiments are SEQUENTIAL, not concurrent.
- Before EACH algorithm, clients are registered to that algorithm's own server.
- config.py is temporarily patched for NUM_ROUNDS and CLIENT_WORKERS,
  then restored automatically, even if the script fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.py"

DEFAULT_ORDER = ["fedavg", "multikrum", "proposed"]

DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "multikrum": "Multi-Krum",
    "proposed": "Proposed",
}

BASE_COMMANDS = {
    "fedavg": [sys.executable, "-m", "experiments.fedAvg.run_selected_round"],
    "multikrum": [sys.executable, "-m", "experiments.multi_Krum.run_selected_round"],
    "proposed": [sys.executable, "main.py"],
}

ALGORITHM_SERVERS = {
    "fedavg": "http://127.0.0.1:8001",
    "multikrum": "http://127.0.0.1:8002",
    "proposed": "http://127.0.0.1:8000",
}

FINAL_PATTERNS = {
    "test_accuracy": re.compile(r"Test Accuracy:\s*([0-9]*\.?[0-9]+)", re.I),
    "test_loss": re.compile(r"Test Loss:\s*([0-9]*\.?[0-9]+)", re.I),
    "macro_f1": re.compile(r"Macro F1:\s*([0-9]*\.?[0-9]+)", re.I),
    "balanced_accuracy": re.compile(r"Balanced Accuracy:\s*([0-9]*\.?[0-9]+)", re.I),
    "worst_class_accuracy": re.compile(r"Worst-class Accuracy:\s*([0-9]*\.?[0-9]+)", re.I),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FedAvg, Multi-Krum and Proposed sequentially and plot comparisons."
    )
    p.add_argument("--rounds", type=int, default=150, help="Maximum rounds. Default: 150")
    p.add_argument("--workers", type=int, default=3, help="CLIENT_WORKERS. Default: 3")
    p.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated order. Default: fedavg,multikrum,proposed",
    )
    p.add_argument(
        "--cooldown",
        type=int,
        default=15,
        help="Seconds to wait between experiments to reduce socket/GPU churn. Default: 15",
    )
    p.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="API server URL. Default: http://127.0.0.1:8000",
    )
    p.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Do not verify that the API server is reachable.",
    )
    p.add_argument(
        "--skip-register",
        action="store_true",
        help="Do not run: python -m client_app.register_all before experiments.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next algorithm if one experiment fails.",
    )
    p.add_argument(
        "--no-patch-config",
        action="store_true",
        help="Do not temporarily patch NUM_ROUNDS/CLIENT_WORKERS in config.py.",
    )
    return p.parse_args()


def banner(text: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{text}\n{line}", flush=True)


def check_python_env() -> None:
    banner("PRE-FLIGHT: PYTHON / CUDA")
    print("Python:", sys.executable)
    try:
        import torch
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("WARNING: CUDA is NOT available in this Python environment.")
    except Exception as exc:
        print("WARNING: Could not import/test torch:", exc)


def check_required_paths() -> None:
    missing = []
    for path in [
        ROOT / "main.py",
        ROOT / "experiments" / "fedAvg",
        ROOT / "experiments" / "multi_Krum",
    ]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Required project paths were not found:\n  - " + "\n  - ".join(missing)
        )


def check_server(url: str, label: str = "API") -> None:
    if requests is None:
        print("WARNING: requests is unavailable; skipping server check.")
        return

    candidates = [
        url.rstrip("/") + "/docs",
        url.rstrip("/") + "/openapi.json",
        url.rstrip("/") + "/",
    ]

    last_error = None
    for target in candidates:
        try:
            r = requests.get(target, timeout=3)
            if r.status_code < 600:
                print(f"{label} server reachable: {target} -> HTTP {r.status_code}")
                return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"{label} server is not reachable at {url}.\n"
        f"Start that server first, then run this file again.\n"
        f"Last error: {last_error}"
    )


def patch_config(rounds: int, workers: int) -> Optional[str]:
    """
    Temporarily force NUM_ROUNDS and CLIENT_WORKERS in config.py.

    This version uses Python's AST so multiline assignments such as:

        CLIENT_WORKERS = _env_int(
            "CLIENT_WORKERS",
            2,
        )

    are replaced as one complete statement instead of only replacing the
    first line (which would leave dangling indented lines and cause an
    IndentationError).

    Returns the original config text so it can be restored in finally.
    """
    if not CONFIG_PATH.exists():
        print("WARNING: config.py not found; relying on CLI/environment values.")
        return None

    import ast

    original = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original, filename=str(CONFIG_PATH))
    except SyntaxError as exc:
        raise RuntimeError(
            f"config.py is already invalid before patching: {exc}"
        ) from exc

    wanted = {
        "NUM_ROUNDS": str(rounds),
        "CLIENT_WORKERS": str(workers),
    }

    spans = {}

    for node in ast.walk(tree):
        target_name = None

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    target_name = target.id
                    break

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in wanted:
                target_name = node.target.id

        if target_name is not None:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            col = getattr(node, "col_offset", 0)
            if start is not None:
                spans[target_name] = (start, end, col)

    if not spans:
        print(
            "WARNING: NUM_ROUNDS / CLIENT_WORKERS assignments were not found; "
            "relying on environment variables."
        )
        return original

    lines = original.splitlines(keepends=True)

    # Replace from bottom to top so line indexes remain valid.
    replacements = []
    for name, value in wanted.items():
        if name in spans:
            start, end, col = spans[name]
            replacements.append((start, end, col, name, value))
        else:
            print(
                f"WARNING: {name} was not found in config.py; "
                f"environment/CLI will be used where supported."
            )

    replacements.sort(key=lambda x: x[0], reverse=True)

    for start, end, col, name, value in replacements:
        indent = " " * col
        newline = "\n"
        if 1 <= start <= len(lines):
            original_line = lines[start - 1]
            if original_line.endswith("\r\n"):
                newline = "\r\n"

        replacement_line = f"{indent}{name} = {value}{newline}"
        lines[start - 1:end] = [replacement_line]
        print(f"Temporary config patch: {name} = {value}")

    updated = "".join(lines)

    # Validate before writing.
    try:
        ast.parse(updated, filename=str(CONFIG_PATH))
    except SyntaxError as exc:
        raise RuntimeError(
            f"Refusing to write patched config.py because it would be invalid: {exc}"
        ) from exc

    CONFIG_PATH.write_text(updated, encoding="utf-8")
    return original


def restore_config(original: Optional[str]) -> None:
    if original is not None and CONFIG_PATH.exists():
        CONFIG_PATH.write_text(original, encoding="utf-8")
        print("\nconfig.py restored to its original contents.")


def run_tee(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    cwd: Path = ROOT,
) -> int:
    """
    Run a child process, stream output to terminal, and save the same output to a log.
    """
    print("\nCOMMAND:", subprocess.list2cmdline(cmd))
    print("LOG:", log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return proc.wait()


def run_registration(
    run_dir: Path,
    env: Dict[str, str],
    algorithm: str,
    server_url: str,
) -> None:
    name = DISPLAY_NAMES[algorithm]
    banner(f"REGISTERING CLIENTS FOR {name} -> {server_url}")
    cmd = [
        sys.executable,
        "-m",
        "client_app.register_all",
        "--server",
        server_url,
    ]
    rc = run_tee(cmd, run_dir / f"register_{algorithm}.log", env)
    if rc != 0:
        raise RuntimeError(
            f"Client registration for {name} failed with exit code {rc}"
        )


def run_experiment(
    key: str,
    rounds: int,
    run_dir: Path,
    env: Dict[str, str],
) -> Tuple[int, Path]:
    name = DISPLAY_NAMES[key]
    banner(f"STARTING {name} | MAX ROUNDS = {rounds}")

    log_path = run_dir / f"{key}.log"
    base = list(BASE_COMMANDS[key])

    # Baseline modules appear to support --rounds.
    # Proposed main.py is controlled by config.py / environment.
    if key in ("fedavg", "multikrum"):
        cmd = base + ["--rounds", str(rounds)]
    else:
        cmd = base

    rc = run_tee(cmd, log_path, env)

    # Compatibility fallback:
    # If FedAvg has an older CLI that does not accept --rounds,
    # config.py has already been patched to 200, so retry without the flag.
    if (
        key == "fedavg"
        and rc != 0
        and log_path.exists()
        and "unrecognized arguments" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
    ):
        print(
            "\nFedAvg CLI does not accept --rounds in this version. "
            "Retrying without --rounds; patched config.py still controls NUM_ROUNDS."
        )
        fallback_log = run_dir / f"{key}_retry.log"
        rc = run_tee(base, fallback_log, env)
        if rc == 0:
            log_path = fallback_log

    return rc, log_path


def parse_log(log_path: Path, algorithm: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    final: Dict[str, float] = {}
    for key, pattern in FINAL_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            final[key] = float(matches[-1])

    round_rows: Dict[int, Dict[str, float]] = {}
    current_round: Optional[int] = None

    header_re = re.compile(r"^\s*ROUND\s+(\d+)(?:\s+SUMMARY)?\s*$", re.I)
    completed_re = re.compile(r"ROUND\s+(\d+)\s+completed", re.I)
    summary_line_re = re.compile(
        r"Round\s+(\d+)\s*\|.*?\bchange=([0-9.eE+-]+)", re.I
    )
    rel_re = re.compile(
        r"(?:Model\s+relative\s+change|Relative\s+change)\s*:\s*([0-9.eE+-]+)",
        re.I,
    )
    val_acc_re = re.compile(r"Validation\s+Accuracy\s*:\s*([0-9.eE+-]+)", re.I)
    global_acc_re = re.compile(r"Global\s+Accuracy\s*:\s*([0-9.eE+-]+)", re.I)

    for raw in text.splitlines():
        line = raw.strip()

        m = header_re.match(line)
        if m:
            current_round = int(m.group(1))
            round_rows.setdefault(current_round, {"round": float(current_round)})
            continue

        m = completed_re.search(line)
        if m:
            completed_round = int(m.group(1))
            round_rows.setdefault(completed_round, {"round": float(completed_round)})
            current_round = completed_round

        m = summary_line_re.search(line)
        if m:
            r = int(m.group(1))
            row = round_rows.setdefault(r, {"round": float(r)})
            row["relative_change"] = float(m.group(2))
            continue

        if current_round is not None:
            m = rel_re.search(line)
            if m:
                round_rows.setdefault(
                    current_round, {"round": float(current_round)}
                )["relative_change"] = float(m.group(1))

            m = val_acc_re.search(line)
            if m:
                round_rows.setdefault(
                    current_round, {"round": float(current_round)}
                )["eval_accuracy"] = float(m.group(1))

            m = global_acc_re.search(line)
            if m and "SKIPPED" not in line.upper():
                round_rows.setdefault(
                    current_round, {"round": float(current_round)}
                )["eval_accuracy"] = float(m.group(1))

    rows = []
    for r in sorted(round_rows):
        row = round_rows[r]
        row["algorithm"] = algorithm
        row["round"] = int(row["round"])
        rows.append(row)

    return final, rows


def write_final_csv(final_results: Dict[str, Dict[str, float]], path: Path) -> None:
    fields = [
        "algorithm",
        "test_accuracy",
        "test_loss",
        "macro_f1",
        "balanced_accuracy",
        "worst_class_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in DEFAULT_ORDER:
            if key not in final_results:
                continue
            row = {"algorithm": DISPLAY_NAMES[key]}
            row.update(final_results[key])
            writer.writerow(row)


def write_round_csv(rows: List[Dict[str, float]], path: Path) -> None:
    fields = ["algorithm", "round", "relative_change", "eval_accuracy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "algorithm": DISPLAY_NAMES.get(str(row["algorithm"]), str(row["algorithm"])),
                "round": row.get("round", ""),
                "relative_change": row.get("relative_change", ""),
                "eval_accuracy": row.get("eval_accuracy", ""),
            }
            writer.writerow(out)


def plot_final_metrics(final_results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping plots.")
        return

    metrics = [
        ("test_accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("worst_class_accuracy", "Worst-class Accuracy"),
    ]
    algos = [k for k in DEFAULT_ORDER if k in final_results]
    if not algos:
        return

    x = list(range(len(metrics)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, algo in enumerate(algos):
        values = [final_results[algo].get(m, float("nan")) * 100.0 for m, _ in metrics]
        positions = [v + (idx - (len(algos) - 1) / 2) * width for v in x]
        ax.bar(positions, values, width=width, label=DISPLAY_NAMES[algo])

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Percent")
    ax.set_title("Final test metrics comparison")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_relative_change(rows: List[Dict[str, float]], out_path: Path) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for algo in DEFAULT_ORDER:
        subset = [
            r for r in rows
            if r.get("algorithm") == algo and "relative_change" in r
        ]
        if not subset:
            continue
        subset.sort(key=lambda x: int(x["round"]))
        ax.plot(
            [int(r["round"]) for r in subset],
            [float(r["relative_change"]) for r in subset],
            label=DISPLAY_NAMES[algo],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Round")
    ax.set_ylabel("Relative model change")
    ax.set_title("Convergence comparison")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_periodic_accuracy(rows: List[Dict[str, float]], out_path: Path) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for algo in DEFAULT_ORDER:
        subset = [
            r for r in rows
            if r.get("algorithm") == algo and "eval_accuracy" in r
        ]
        if not subset:
            continue
        subset.sort(key=lambda x: int(x["round"]))
        ax.plot(
            [int(r["round"]) for r in subset],
            [float(r["eval_accuracy"]) * 100.0 for r in subset],
            marker="o",
            markersize=3,
            label=DISPLAY_NAMES[algo],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Periodic validation/global evaluation accuracy")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(
    final_results: Dict[str, Dict[str, float]],
    statuses: Dict[str, int],
    run_dir: Path,
    rounds: int,
    workers: int,
) -> None:
    lines = []
    lines.append("TrustFed-IoT automated comparison")
    lines.append(f"Maximum rounds: {rounds}")
    lines.append(f"CLIENT_WORKERS: {workers}")
    lines.append("")
    lines.append("Final results:")
    for algo in DEFAULT_ORDER:
        if algo not in statuses:
            continue
        name = DISPLAY_NAMES[algo]
        lines.append(f"\n{name} | exit_code={statuses[algo]}")
        metrics = final_results.get(algo, {})
        for k in [
            "test_accuracy",
            "test_loss",
            "macro_f1",
            "balanced_accuracy",
            "worst_class_accuracy",
        ]:
            if k in metrics:
                lines.append(f"  {k}: {metrics[k]}")
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    order = [x.strip().lower() for x in args.order.split(",") if x.strip()]
    unknown = [x for x in order if x not in BASE_COMMANDS]
    if unknown:
        raise ValueError(f"Unknown algorithms in --order: {unknown}")

    check_required_paths()
    check_python_env()

    if not args.skip_server_check:
        banner("PRE-FLIGHT: CHECKING ALL THREE SERVERS")
        for algo in DEFAULT_ORDER:
            check_server(ALGORITHM_SERVERS[algo], DISPLAY_NAMES[algo])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "comparison_runs" / f"run_{stamp}_r{args.rounds}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NUM_ROUNDS"] = str(args.rounds)
    env["CLIENT_WORKERS"] = str(args.workers)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "root": str(ROOT),
        "rounds": args.rounds,
        "workers": args.workers,
        "order": order,
        "servers": ALGORITHM_SERVERS,
        "commands": {
            k: subprocess.list2cmdline(BASE_COMMANDS[k]) for k in order
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    original_config: Optional[str] = None
    statuses: Dict[str, int] = {}
    log_paths: Dict[str, Path] = {}

    try:
        if not args.no_patch_config:
            original_config = patch_config(args.rounds, args.workers)

        for idx, algo in enumerate(order):
            if not args.skip_register:
                run_registration(
                    run_dir=run_dir,
                    env=env,
                    algorithm=algo,
                    server_url=ALGORITHM_SERVERS[algo],
                )

            rc, log_path = run_experiment(
                key=algo,
                rounds=args.rounds,
                run_dir=run_dir,
                env=env,
            )
            statuses[algo] = rc
            log_paths[algo] = log_path

            if rc != 0:
                print(f"\nERROR: {DISPLAY_NAMES[algo]} exited with code {rc}.")
                if not args.continue_on_error:
                    print("Stopping so incomplete results are not compared as if they were complete.")
                    break

            if idx < len(order) - 1 and args.cooldown > 0:
                print(
                    f"\nCooldown: waiting {args.cooldown} seconds before the next experiment "
                    "(helps reduce Windows socket/GPU churn)..."
                )
                time.sleep(args.cooldown)

    finally:
        restore_config(original_config)

    banner("PARSING RESULTS AND CREATING CHARTS")

    final_results: Dict[str, Dict[str, float]] = {}
    all_round_rows: List[Dict[str, float]] = []

    for algo, log_path in log_paths.items():
        if not log_path.exists():
            continue
        final, rows = parse_log(log_path, algo)
        final_results[algo] = final
        all_round_rows.extend(rows)

    write_final_csv(final_results, run_dir / "final_metrics.csv")
    write_round_csv(all_round_rows, run_dir / "round_metrics.csv")
    plot_final_metrics(final_results, run_dir / "final_metrics_comparison.png")
    plot_relative_change(all_round_rows, run_dir / "relative_change_comparison.png")
    plot_periodic_accuracy(all_round_rows, run_dir / "periodic_accuracy_comparison.png")
    write_summary(
        final_results=final_results,
        statuses=statuses,
        run_dir=run_dir,
        rounds=args.rounds,
        workers=args.workers,
    )

    print("\nDONE.")
    print("Results folder:", run_dir)
    print("Important files:")
    print("  - final_metrics.csv")
    print("  - round_metrics.csv")
    print("  - final_metrics_comparison.png")
    print("  - relative_change_comparison.png")
    print("  - periodic_accuracy_comparison.png (if evaluation points exist)")
    print("  - summary.txt")
    print("  - fedavg.log / multikrum.log / proposed.log")

    failed = [k for k, rc in statuses.items() if rc != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
