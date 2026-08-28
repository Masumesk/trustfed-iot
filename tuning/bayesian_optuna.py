
import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure project root and tuning/ are on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import optuna
except ImportError:
    print("optuna is not installed.")
    print("Install it with: pip install optuna")
    sys.exit(1)

import parameter as sweep
from parameter import (
    read_csv_rows,
    safe_float,
    run_single_experiment,
    summarize_run,
    stop_process,
    wait_for_server,
)


# ── Search space definition ──────────────────────────────────────────────────

SEARCH_SPACE = {
    "LAMBDA_TRUST": {
        "type": "float",
        "low": 0.1,
        "high": 0.9,
        "step": 0.05,
    },
    "TRUST_THRESHOLD": {
        "type": "float",
        "low": 0.2,
        "high": 0.8,
        "step": 0.05,
    },
    "T_NEAR": {
        "type": "float",
        "low": 0.4,
        "high": 0.95,
        "step": 0.05,
    },
    "SELECTION_ALPHA": {
        "type": "float",
        "low": 0.1,
        "high": 0.9,
        "step": 0.05,
    },
    "TRIM_RATIO": {
        "type": "float",
        "low": 0.0,
        "high": 0.4,
        "step": 0.05,
    },
    "OPTICS_XI": {
        "type": "float",
        "low": 0.02,
        "high": 0.15,
        "step": 0.01,
    },
    "OPTICS_MIN_SAMPLES": {
        "type": "int",
        "low": 2,
        "high": 6,
    },
    "NOISE_ASSIGNMENT_THRESHOLD": {
        "type": "float",
        "low": 0.3,
        "high": 0.9,
        "step": 0.05,
    },
    "RANDOM_RATIO": {
        "type": "float",
        "low": 0.2,
        "high": 0.8,
        "step": 0.05,
    },
}


def suggest_params(trial):
    """ suggest parameter values from the search space """
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
            )
    return params


def _is_port_in_use(port, host="127.0.0.1"):
    """ Check if a port is in use. """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _kill_port(port):
    """ Kill any process using the given port (Windows + Linux). """
    import subprocess as _sp
    import platform

    if platform.system() == "Windows":
        out = _sp.run(["netstat", "-ano", f":{port}"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "LISTENING" in line:
                _sp.run(["taskkill", "/F", "/PID", line.strip().split()[-1]], capture_output=True)
    else:
        out = _sp.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True).stdout
        for pid in out.strip().split():
            _sp.run(["kill", "-9", pid], capture_output=True)


def run_trial(project_dir, params, run_dir, metric):
    """
    Run one experiment with all param overrides and return metric value.
    Sets ALL params as env vars, then delegates to parameter.run_single_experiment.
    """
    import time

    # Set all params as env vars before calling run_single_experiment
    for k, v in params.items():
        os.environ[k] = str(v)

    # Kill port if occupied
    if _is_port_in_use(8000):
        print("  Port 8000 in use, killing...")
        _kill_port(8000)
        time.sleep(2)

    try:
        # run_single_experiment sets env[parameter]=value and runs server/register/main
        # We pass LAMBDA_TRUST as the parameter — all other params are already in env
        result_csv = run_single_experiment(
            project_dir,
            "LAMBDA_TRUST",
            params.get("LAMBDA_TRUST", 0.5),
            run_dir,
        )

        summary = summarize_run(
            result_csv, "LAMBDA_TRUST", params.get("LAMBDA_TRUST", 0.5)
        )

        metric_map = {
            "mean_last_5_accuracy": summary.get("mean_last_5_accuracy"),
            "final_accuracy": summary.get("final_accuracy"),
            "best_accuracy": summary.get("best_accuracy"),
            "final_loss": -summary.get("final_loss", 0),
            "final_macro_f1": summary.get("final_macro_f1"),
        }

        return metric_map.get(metric) or float("inf")

    except Exception as e:
        import traceback
        error_log = run_dir / "error.log"
        with open(error_log, "w", encoding="utf-8") as ef:
            ef.write(f"Error: {e}\n")
            ef.write(traceback.format_exc())
        print(f"  Trial FAILED: {e}")
        return float("inf")


def objective(trial, project_dir, metric, output_dir):
    """ Optuna objective function. """
    params = suggest_params(trial)

    trial_dir = output_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    value = run_trial(project_dir, params, trial_dir, metric)

    # save trial result
    result_path = trial_dir / "result.csv"
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": metric, "value": value})

    print(f"  Result: {metric}={value:.6f}")

    return value


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for TrustFed (Optuna)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Number of optimization trials (default: 30)",
    )
    parser.add_argument(
        "--metric", type=str, default="mean_last_5_accuracy",
        help="Metric to optimize (default: mean_last_5_accuracy)",
    )
    parser.add_argument(
        "--project-dir", type=str, default=".",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing study database to resume",
    )
    parser.add_argument(
        "--sampler", type=str, default="tpe",
        choices=["tpe", "random", "cmaes"],
        help="Sampling algorithm (default: tpe)",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not (project_dir / "main.py").exists():
        raise FileNotFoundError(
            f"main.py not found in {project_dir}"
        )

    output_dir = project_dir / "results" / "bayesian_optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    # study database
    db_path = args.resume or str(output_dir / "study_trustfed.db")
    storage = f"sqlite:///{db_path}"

    # sampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=42)

    # create or load study
    study = optuna.create_study(
        study_name="trustfed_optimization",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        direction="maximize",
    )

    print(f"\nSearch space:")
    for name, spec in SEARCH_SPACE.items():
        print(f"  {name}: [{spec['low']}, {spec['high']}]")
    print(f"\nMetric: {args.metric}")
    print(f"Trials: {args.n_trials}")
    print(f"Sampler: {args.sampler}")
    print(f"Database: {db_path}")
    print(f"Completed trials so far: {len(study.trials)}")
    print(f"\n⚠️  Tuning on VALIDATION set (no data leakage)")
    print(f"   Use /evaluate_final for test set after selecting params")

    # run optimization
    study.optimize(
        lambda trial: objective(
            trial, project_dir, args.metric, output_dir
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── Save results ──
    best = study.best_trial
    print(f"\n{'='*60}\nBEST TRIAL #{best.number} | {args.metric}={best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # best_params.csv
    with open(output_dir / "best_params.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "value"])
        w.writeheader()
        for k, v in best.params.items():
            w.writerow({"parameter": k, "value": v})

    # all_trials.csv
    fields = ["trial", "value"] + list(SEARCH_SPACE.keys())
    with open(output_dir / "all_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in study.trials:
            row = {"trial": t.number, "value": t.value, **t.params}
            w.writerow(row)

    # summary.csv
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["info", "value"])
        w.writeheader()
        w.writerow({"info": "best_metric", "value": best.value})
        w.writerow({"info": "best_trial", "value": best.number})
        w.writerow({"info": "total_trials", "value": len(study.trials)})
        for k, v in best.params.items():
            w.writerow({"info": f"best_{k}", "value": v})

    print(f"\nResults: {output_dir}")

    # top 5
    top5 = sorted(study.trials, key=lambda t: t.value or float("inf"), reverse=True)[:5]
    print("Top 5:")
    for t in top5:
        print(f"  #{t.number}: {args.metric}={t.value:.6f}")


if __name__ == "__main__":
    main()
