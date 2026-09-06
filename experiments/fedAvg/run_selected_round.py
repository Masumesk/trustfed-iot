import argparse

from config import NUM_ROUNDS
from experiments.common_round_runner import FEDAVG, run_experiment


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        type=str,
        default="http://127.0.0.1:8001",
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
        method=FEDAVG,
        method_name="FedAvg",
        round_csv_path="results/fedavg.csv",
        final_csv_path="results/fedavg_final_test.csv",
    )


if __name__ == "__main__":
    main()
