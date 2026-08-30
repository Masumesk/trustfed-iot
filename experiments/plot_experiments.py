"""Aggregate benchmark CSVs and generate thesis-ready comparison outputs."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["proposed", "fedavg", "multikrum"]
METHOD_LABELS = {
    "proposed": "Proposed",
    "fedavg": "FedAvg",
    "multikrum": "Multi-Krum",
}
ATTACK_ORDER = ["clean", "gaussian", "sign_flip", "scaling", "label_flip"]
ATTACK_TITLES = {
    "clean": "No Attack",
    "gaussian": "Gaussian Noise Attack",
    "sign_flip": "Sign Flip Attack",
    "scaling": "Scaling Attack",
    "label_flip": "Label Flip Attack",
}


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "skipped"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_round_csv(path: Path) -> List[Tuple[int, float]]:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            round_value = parse_float(row.get("round"))
            accuracy = parse_float(row.get("val_accuracy"))
            if accuracy is None:
                accuracy = parse_float(row.get("accuracy"))

            if round_value is None or accuracy is None:
                continue

            rows.append((int(round_value), float(accuracy)))

    return rows


def discover_attacks(root: Path) -> List[str]:
    raw_root = root / "raw"
    if not raw_root.exists():
        return []

    existing = [p.name for p in raw_root.iterdir() if p.is_dir()]
    ordered = [a for a in ATTACK_ORDER if a in existing]
    ordered.extend(sorted(a for a in existing if a not in ATTACK_ORDER))
    return ordered


def discover_methods(root: Path, attack: str) -> List[str]:
    attack_root = root / "raw" / attack
    if not attack_root.exists():
        return []

    existing = [p.name for p in attack_root.iterdir() if p.is_dir()]
    ordered = [m for m in METHOD_ORDER if m in existing]
    ordered.extend(sorted(m for m in existing if m not in METHOD_ORDER))
    return ordered


def aggregate_rounds(files: Iterable[Path]):
    by_round: Dict[int, List[float]] = defaultdict(list)

    for path in files:
        for round_id, accuracy in read_round_csv(path):
            by_round[round_id].append(accuracy)

    result = []
    for round_id in sorted(by_round):
        values = np.asarray(by_round[round_id], dtype=float)
        result.append(
            {
                "round": round_id,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
        )

    return result


def save_per_round_aggregate(root: Path, attacks: List[str]) -> Path:
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output = summary_dir / "accuracy_per_round_aggregated.csv"

    with output.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "attack",
            "method",
            "round",
            "accuracy_mean_pct",
            "accuracy_std_pct",
            "num_runs",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for attack in attacks:
            for method in discover_methods(root, attack):
                files = sorted((root / "raw" / attack / method).glob("seed_*.csv"))
                for item in aggregate_rounds(files):
                    writer.writerow(
                        {
                            "attack": attack,
                            "method": method,
                            "round": item["round"],
                            "accuracy_mean_pct": item["mean"] * 100.0,
                            "accuracy_std_pct": item["std"] * 100.0,
                            "num_runs": item["n"],
                        }
                    )

    return output


def plot_attack(root: Path, attack: str) -> None:
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    plotted = False

    for method in discover_methods(root, attack):
        files = sorted((root / "raw" / attack / method).glob("seed_*.csv"))
        aggregated = aggregate_rounds(files)
        if not aggregated:
            continue

        rounds = np.asarray([item["round"] for item in aggregated], dtype=int)
        means = np.asarray([item["mean"] for item in aggregated], dtype=float) * 100.0
        stds = np.asarray([item["std"] for item in aggregated], dtype=float) * 100.0

        line = ax.plot(
            rounds,
            means,
            linewidth=2,
            label=METHOD_LABELS.get(method, method),
        )[0]

        if np.any(stds > 0):
            ax.fill_between(
                rounds,
                means - stds,
                means + stds,
                alpha=0.15,
                color=line.get_color(),
            )

        plotted = True

    if not plotted:
        plt.close(fig)
        return

    title = ATTACK_TITLES.get(attack, attack.replace("_", " ").title())
    ax.set_title(f"Accuracy per Round — {title}")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    png_path = figures_dir / f"accuracy_per_round_{attack}.png"
    pdf_path = figures_dir / f"accuracy_per_round_{attack}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def mean_std(values: List[float]):
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def last_k_accuracy(path: Path, last_k: int):
    values = [accuracy for _, accuracy in read_round_csv(path)]
    if not values:
        return None
    return float(np.mean(values[-last_k:]))


def build_final_summary(root: Path, attacks: List[str], last_k: int) -> Path:
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output = summary_dir / "final_comparison.csv"

    fields = [
        "attack",
        "method",
        "num_runs",
        f"last_{last_k}_val_accuracy_mean_pct",
        f"last_{last_k}_val_accuracy_std_pct",
        "test_accuracy_mean_pct",
        "test_accuracy_std_pct",
        "test_loss_mean",
        "test_loss_std",
        "macro_f1_mean_pct",
        "macro_f1_std_pct",
        "balanced_accuracy_mean_pct",
        "balanced_accuracy_std_pct",
        "worst_class_accuracy_mean_pct",
        "worst_class_accuracy_std_pct",
    ]

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for attack in attacks:
            for method in discover_methods(root, attack):
                raw_files = sorted((root / "raw" / attack / method).glob("seed_*.csv"))
                final_files = sorted((root / "final" / attack / method).glob("seed_*.json"))

                last_k_values = [
                    value
                    for value in (last_k_accuracy(path, last_k) for path in raw_files)
                    if value is not None
                ]

                metrics = defaultdict(list)
                for path in final_files:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for key in [
                        "accuracy",
                        "loss",
                        "macro_f1",
                        "balanced_accuracy",
                        "worst_class_accuracy",
                    ]:
                        value = parse_float(payload.get(key))
                        if value is not None:
                            metrics[key].append(value)

                last_mean, last_std = mean_std(last_k_values)
                test_mean, test_std = mean_std(metrics["accuracy"])
                loss_mean, loss_std = mean_std(metrics["loss"])
                f1_mean, f1_std = mean_std(metrics["macro_f1"])
                bal_mean, bal_std = mean_std(metrics["balanced_accuracy"])
                worst_mean, worst_std = mean_std(metrics["worst_class_accuracy"])

                def pct(value):
                    return "" if value is None else value * 100.0

                writer.writerow(
                    {
                        "attack": attack,
                        "method": method,
                        "num_runs": len(final_files),
                        f"last_{last_k}_val_accuracy_mean_pct": pct(last_mean),
                        f"last_{last_k}_val_accuracy_std_pct": pct(last_std),
                        "test_accuracy_mean_pct": pct(test_mean),
                        "test_accuracy_std_pct": pct(test_std),
                        "test_loss_mean": "" if loss_mean is None else loss_mean,
                        "test_loss_std": "" if loss_std is None else loss_std,
                        "macro_f1_mean_pct": pct(f1_mean),
                        "macro_f1_std_pct": pct(f1_std),
                        "balanced_accuracy_mean_pct": pct(bal_mean),
                        "balanced_accuracy_std_pct": pct(bal_std),
                        "worst_class_accuracy_mean_pct": pct(worst_mean),
                        "worst_class_accuracy_std_pct": pct(worst_std),
                    }
                )

    return output



def import_result_zip(zip_path: Path, root: Path) -> None:
    """Merge a shard export ZIP into an existing benchmark directory.

    The shard ZIP contains paths relative to results/benchmark, so two
    independent machines can be merged simply by extracting the missing shard
    into the same root. Existing identical filenames are overwritten; shards
    created by the runner are non-overlapping by construction.
    """
    zip_path = zip_path.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Shard ZIP not found: {zip_path}")

    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path inside ZIP: {member.filename}")
        archive.extractall(root)

    print(f"Imported shard: {zip_path}")


def check_seed_coverage(root: Path) -> None:
    """Print a compact coverage report before plotting."""
    attacks = discover_attacks(root)
    if not attacks:
        return

    print("\nResult coverage:")
    for attack in attacks:
        parts = []
        for method in discover_methods(root, attack):
            seeds = sorted(
                int(path.stem.split("_")[-1])
                for path in (root / "raw" / attack / method).glob("seed_*.csv")
                if path.stem.split("_")[-1].isdigit()
            )
            parts.append(f"{method}={seeds}")
        print(f"  {attack}: " + ", ".join(parts))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/benchmark"),
        help="Benchmark result directory.",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=10,
        help="Number of final validation rounds averaged in the summary.",
    )
    parser.add_argument(
        "--import-zip",
        type=Path,
        action="append",
        default=[],
        help=(
            "Shard export ZIP to merge into --root before plotting. "
            "May be passed multiple times."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()

    if args.last_k < 1:
        raise ValueError("--last-k must be >= 1")

    for zip_path in args.import_zip:
        import_result_zip(zip_path, root)

    check_seed_coverage(root)

    attacks = discover_attacks(root)
    if not attacks:
        raise FileNotFoundError(f"No benchmark CSVs found under {root / 'raw'}")

    aggregate_path = save_per_round_aggregate(root, attacks)
    print(f"Saved: {aggregate_path}")

    for attack in attacks:
        plot_attack(root, attack)

    final_path = build_final_summary(root, attacks, args.last_k)
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
