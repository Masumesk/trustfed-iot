"""Automate TrustFed-IoT comparison experiments.

This runner supports deterministic sharding so the same Git checkout can be
used on completely independent machines. Results are kept out of Git and can
be exported as a small ZIP for manual transfer and final aggregation.

Typical two-machine usage:

Machine 1:
    python -m experiments.run_all_experiments --num-shards 2 --shard-index 0 --export-zip

Machine 2:
    python -m experiments.run_all_experiments --num-shards 2 --shard-index 1 --export-zip

Each machine receives a deterministic, non-overlapping subset of all requested
(attack, seed, method) jobs. The ZIP contains only raw/final benchmark outputs
and a manifest; logs stay local.
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
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "results" / "benchmark"
EXPORT_ROOT = ROOT / "results" / "exports"


METHODS: Dict[str, dict] = {
    "proposed": {
        "display": "Proposed",
        "port": 8000,
        "server_cmd": [sys.executable, "-m", "server_app.run_server"],
        "run_cmd": [sys.executable, "main.py"],
        "source_csv": ROOT / "results" / "proposed.csv",
    },
    "fedavg": {
        "display": "FedAvg",
        "port": 8001,
        "server_cmd": [sys.executable, "-m", "experiments.fedAvg.server.run_server"],
        "run_cmd": [sys.executable, "-m", "experiments.fedAvg.run_selected_round"],
        "source_csv": ROOT / "results" / "fedavg.csv",
    },
    "multikrum": {
        "display": "Multi-Krum",
        "port": 8002,
        "server_cmd": [sys.executable, "-m", "experiments.multi_Krum.server.run_server"],
        "run_cmd": [sys.executable, "-m", "experiments.multi_Krum.run_selected_round"],
        "source_csv": ROOT / "results" / "multikrum.csv",
    },
}

ATTACKS = ["clean", "gaussian", "sign_flip", "scaling", "label_flip"]
Job = Tuple[str, int, str]  # (attack, seed, method)


def wait_for_server(url: str, process: subprocess.Popen, log_path: Path, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            except OSError:
                pass
            raise RuntimeError(
                f"Server exited before becoming ready.\nLog: {log_path}\n\n{tail}"
            )

        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc

        time.sleep(1)

    raise TimeoutError(
        f"Server did not become ready within {timeout}s at {url}. "
        f"Last error: {last_error}. See {log_path}"
    )


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_logged(cmd: List[str], env: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"Command failed with code {completed.returncode}: {' '.join(cmd)}\n"
            f"Log: {log_path}\n\n{tail}"
        )


def make_env(
    *,
    attack: str,
    seed: int,
    rounds: int,
    malicious_ratio: float,
    data_seed: int,
    server_url: str,
) -> dict:
    env = os.environ.copy()

    # Keep the data partition identical for all methods and all replicate seeds.
    env["DATA_SEED"] = str(data_seed)

    # Replicate-specific randomness; identical across methods for a fair comparison.
    env["MODEL_SEED"] = str(seed)
    env["TRAINING_SEED"] = str(seed)
    env["SELECTION_SEED"] = str(seed)
    env["MALICIOUS_SEED"] = str(seed)

    env["NUM_ROUNDS"] = str(rounds)
    env["SERVER_URL"] = server_url

    # Accuracy-per-round benchmark: evaluate every round and do not early-stop.
    env["MODEL_CHANGE_THRESHOLD"] = "1000000000"
    env["PATIENCE"] = str(rounds + 1000)

    if attack == "clean":
        env.pop("ATTACK_TYPE", None)
        env["MALICIOUS_RATIO"] = "0.0"
    else:
        env["ATTACK_TYPE"] = attack
        env["MALICIOUS_RATIO"] = str(malicious_ratio)

    return env


def clear_legacy_outputs(method: str) -> None:
    candidates = [
        METHODS[method]["source_csv"],
        ROOT / "results" / f"{method}_final_test.csv",
    ]

    if method == "proposed":
        candidates.append(ROOT / "results" / "proposed_final_test.csv")
    elif method == "multikrum":
        candidates.append(ROOT / "results" / "multikrum_final_test.csv")

    for path in candidates:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _parse_float(text: str):
    text = text.strip()
    if text.upper() == "SKIPPED" or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def recover_round_csv_from_log(log_path: Path, target: Path, method: str) -> bool:
    """Best-effort fallback when a baseline did not emit its legacy CSV.

    This keeps the automation usable across slightly different project revisions.
    It intentionally extracts only the fields needed by plot_experiments.py.
    """
    if not log_path.exists():
        return False

    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []

    # FedAvg/Multi-Krum print one compact summary line per round at the end.
    summary_pattern = re.compile(
        r"Round\s+(\d+)\s*\|\s*val_accuracy=([^|]+)\|\s*"
        r"val_loss=([^|]+)\|\s*change=([^|]+)\|"
    )
    for match in summary_pattern.finditer(text):
        round_id = int(match.group(1))
        accuracy = _parse_float(match.group(2))
        val_loss = _parse_float(match.group(3))
        change = _parse_float(match.group(4))
        if accuracy is not None:
            rows.append(
                {
                    "round": round_id,
                    "val_accuracy": accuracy,
                    "val_loss": "" if val_loss is None else val_loss,
                    "relative_change": "" if change is None else change,
                }
            )

    # Proposed prints Global Accuracy/Loss inside each round instead of a summary.
    if not rows and method == "proposed":
        current_round = None
        current_change = None
        current_acc = None
        current_loss = None
        for line in text.splitlines():
            m = re.match(r"\s*ROUND\s+(\d+)\s*$", line)
            if m:
                if current_round is not None and current_acc is not None:
                    rows.append(
                        {
                            "round": current_round,
                            "val_accuracy": current_acc,
                            "val_loss": "" if current_loss is None else current_loss,
                            "relative_change": "" if current_change is None else current_change,
                        }
                    )
                current_round = int(m.group(1))
                current_change = current_acc = current_loss = None
                continue

            m = re.search(r"Model relative change:\s*([0-9.eE+-]+)", line)
            if m:
                current_change = float(m.group(1))
                continue

            m = re.search(r"Global Accuracy:\s*([0-9.eE+-]+)", line)
            if m:
                current_acc = float(m.group(1))
                continue

            m = re.search(r"Global Loss:\s*([0-9.eE+-]+)", line)
            if m:
                current_loss = float(m.group(1))

        if current_round is not None and current_acc is not None:
            rows.append(
                {
                    "round": current_round,
                    "val_accuracy": current_acc,
                    "val_loss": "" if current_loss is None else current_loss,
                    "relative_change": "" if current_change is None else current_change,
                }
            )

    if not rows:
        return False

    # Remove duplicate round rows if a log contains repeated summaries.
    by_round = {int(row["round"]): row for row in rows}
    rows = [by_round[k] for k in sorted(by_round)]

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["round", "val_accuracy", "val_loss", "relative_change"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WARN] Legacy CSV missing; recovered {len(rows)} rounds from {log_path.name}")
    return True


def run_single(
    *,
    method: str,
    attack: str,
    seed: int,
    rounds: int,
    malicious_ratio: float,
    data_seed: int,
    overwrite: bool,
) -> None:
    spec = METHODS[method]
    port = spec["port"]
    server_url = f"http://127.0.0.1:{port}"

    raw_dir = BENCHMARK_ROOT / "raw" / attack / method
    final_dir = BENCHMARK_ROOT / "final" / attack / method
    log_dir = BENCHMARK_ROOT / "logs" / attack / method

    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_target = raw_dir / f"seed_{seed}.csv"
    final_target = final_dir / f"seed_{seed}.json"

    if raw_target.exists() and final_target.exists() and not overwrite:
        print(
            f"[SKIP] attack={attack:10s} method={method:10s} seed={seed} "
            f"(outputs already exist)"
        )
        return

    env = make_env(
        attack=attack,
        seed=seed,
        rounds=rounds,
        malicious_ratio=malicious_ratio,
        data_seed=data_seed,
        server_url=server_url,
    )

    clear_legacy_outputs(method)

    server_log_path = log_dir / f"seed_{seed}_server.log"
    register_log_path = log_dir / f"seed_{seed}_register.log"
    run_log_path = log_dir / f"seed_{seed}_run.log"

    print(
        f"[RUN ] attack={attack:10s} method={method:10s} seed={seed} rounds={rounds}"
    )

    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_log = server_log_path.open("w", encoding="utf-8")
    server_process = None

    try:
        server_process = subprocess.Popen(
            spec["server_cmd"],
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for_server(server_url, server_process, server_log_path)

        run_logged(
            [
                sys.executable,
                "-m",
                "client_app.register_all",
                "--server",
                server_url,
            ],
            env,
            register_log_path,
        )

        run_cmd = list(spec["run_cmd"])
        if method in ("fedavg", "multikrum"):
            run_cmd.extend(["--server", server_url, "--rounds", str(rounds)])

        run_logged(run_cmd, env, run_log_path)

        source_csv = Path(spec["source_csv"])
        if source_csv.exists():
            shutil.copy2(source_csv, raw_target)
        elif not recover_round_csv_from_log(run_log_path, raw_target, method):
            raise FileNotFoundError(
                f"Expected per-round CSV was not created: {source_csv}\n"
                f"Also could not recover round metrics from: {run_log_path}"
            )

        # Always evaluate final TEST here, independent of each method's own
        # convergence/early-stopping behavior.
        response = requests.get(f"{server_url}/evaluate_final", timeout=300)
        response.raise_for_status()
        final_result = response.json()

        final_payload = {
            "attack": attack,
            "method": method,
            "seed": seed,
            "rounds": rounds,
            "malicious_ratio": 0.0 if attack == "clean" else malicious_ratio,
            "data_seed": data_seed,
            **final_result,
        }

        final_target.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[DONE] attack={attack:10s} method={method:10s} seed={seed} "
            f"test_acc={float(final_result['accuracy']) * 100:.2f}%"
        )

    finally:
        if server_process is not None:
            terminate_process(server_process)
        server_log.close()


def build_jobs(attacks: Sequence[str], seeds: Sequence[int], methods: Sequence[str]) -> List[Job]:
    # Deterministic ordering on every machine and OS.
    return [
        (attack, seed, method)
        for attack in attacks
        for seed in seeds
        for method in methods
    ]


def select_shard(jobs: Sequence[Job], shard_index: int, num_shards: int) -> List[Job]:
    """Shard by comparison group (attack, seed), not by individual method.

    Keeping Proposed/FedAvg/Multi-Krum for the same attack+seed on the same
    machine reduces the chance that hardware/runtime differences become a
    confounder in a paired comparison.
    """
    group_order = []
    grouped = {}
    for job in jobs:
        attack, seed, _method = job
        key = (attack, seed)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(job)

    selected = []
    for group_index, key in enumerate(group_order):
        if group_index % num_shards == shard_index:
            selected.extend(grouped[key])
    return selected


def write_manifest(
    *,
    all_jobs: Sequence[Job],
    shard_jobs: Sequence[Job],
    shard_index: int,
    num_shards: int,
    rounds: int,
    malicious_ratio: float,
    data_seed: int,
) -> Path:
    manifests = BENCHMARK_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    path = manifests / f"shard_{shard_index}_of_{num_shards}.json"

    payload = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rounds": rounds,
        "malicious_ratio": malicious_ratio,
        "data_seed": data_seed,
        "total_jobs": len(all_jobs),
        "jobs_in_shard": len(shard_jobs),
        "jobs": [
            {"attack": attack, "seed": seed, "method": method}
            for attack, seed, method in shard_jobs
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_shard_zip(
    *,
    jobs: Sequence[Job],
    shard_index: int,
    num_shards: int,
    manifest_path: Path,
) -> Path:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = EXPORT_ROOT / f"benchmark_shard_{shard_index}_of_{num_shards}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if manifest_path.exists():
            archive.write(
                manifest_path,
                manifest_path.relative_to(BENCHMARK_ROOT).as_posix(),
            )

        for attack, seed, method in jobs:
            for relative in [
                Path("raw") / attack / method / f"seed_{seed}.csv",
                Path("final") / attack / method / f"seed_{seed}.json",
            ]:
                full_path = BENCHMARK_ROOT / relative
                if full_path.exists():
                    archive.write(full_path, relative.as_posix())

    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrustFed-IoT benchmark experiments automatically."
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=ATTACKS,
        default=ATTACKS,
        help="Attack scenarios to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Methods to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Replicate seeds. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Fixed FL rounds for every run.",
    )
    parser.add_argument(
        "--malicious-ratio",
        type=float,
        default=0.2,
        help="Malicious client ratio for attack scenarios.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Fixed non-IID partition seed shared by all runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run combinations whose result files already exist.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate plots/summary after experiments.",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=10,
        help="Number of final validation rounds averaged in summary table.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the requested job list into N deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to run. Use 0 on machine 1, 1 on machine 2.",
    )
    parser.add_argument(
        "--export-zip",
        action="store_true",
        help="Export this shard's raw/final outputs to results/exports/*.zip.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if not (0.0 <= args.malicious_ratio < 1.0):
        raise ValueError("--malicious-ratio must be in [0, 1)")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)

    all_jobs = build_jobs(args.attacks, args.seeds, args.methods)
    shard_jobs = select_shard(all_jobs, args.shard_index, args.num_shards)

    manifest_path = write_manifest(
        all_jobs=all_jobs,
        shard_jobs=shard_jobs,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        rounds=args.rounds,
        malicious_ratio=args.malicious_ratio,
        data_seed=args.data_seed,
    )

    print(f"Total requested jobs: {len(all_jobs)}")
    print(
        f"Shard {args.shard_index}/{args.num_shards - 1}: "
        f"{len(shard_jobs)} jobs"
    )
    print(f"Output root: {BENCHMARK_ROOT}")
    print(f"Manifest: {manifest_path}")

    for i, (attack, seed, method) in enumerate(shard_jobs, start=1):
        print(
            f"\nShard progress {i}/{len(shard_jobs)} | "
            f"global job subset: attack={attack}, seed={seed}, method={method}"
        )
        run_single(
            method=method,
            attack=attack,
            seed=seed,
            rounds=args.rounds,
            malicious_ratio=args.malicious_ratio,
            data_seed=args.data_seed,
            overwrite=args.overwrite,
        )

    # Partial shards should not generate thesis plots because the comparison is
    # incomplete until the other shard ZIP has been imported.
    if args.num_shards == 1 and not args.no_plots:
        print("Generating Accuracy-per-Round figures and final summary...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.plot_experiments",
                "--root",
                str(BENCHMARK_ROOT),
                "--last-k",
                str(args.last_k),
            ],
            cwd=ROOT,
            check=True,
        )
    elif args.num_shards > 1 and not args.no_plots:
        print(
            "Plots skipped because this is only one shard. "
            "Import the other shard on one machine, then run plot_experiments."
        )

    if args.export_zip:
        zip_path = export_shard_zip(
            jobs=shard_jobs,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            manifest_path=manifest_path,
        )
        print(f"Shard export: {zip_path}")

    print("\nAll jobs assigned to this shard are complete.")


if __name__ == "__main__":
    main()
