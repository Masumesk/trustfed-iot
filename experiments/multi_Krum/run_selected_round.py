import argparse

from config import NUM_ROUNDS
from experiments.common_round_runner import MULTI_KRUM, run_experiment


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

    run_experiment(
        server_url=args.server,
        num_rounds=args.rounds,
        method=MULTI_KRUM,
        method_name="Multi-Krum",
        round_csv_path="results/multikrum.csv",
        final_csv_path="results/multikrum_final_test.csv",
    )


if __name__ == "__main__":
    main()