import argparse
import subprocess
import sys

from config import NUM_CLIENTS


parser = argparse.ArgumentParser()

parser.add_argument(
    "--server",
    type=str,
    default="http://127.0.0.1:8000"
)

args = parser.parse_args()


for client_id in range(NUM_CLIENTS):

    print(
        f"Registering client {client_id}"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "client_app.register_client",

            "--id",
            str(client_id),

            "--server",
            args.server
        ],
        check=True
    )


print("All clients registered")