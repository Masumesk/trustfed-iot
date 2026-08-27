import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from config import NUM_CLIENTS, SERVER_URL


parser = argparse.ArgumentParser()

parser.add_argument(
    "--server",
    type=str,
    default=SERVER_URL
)

args = parser.parse_args()


def register_one(client_id):
    print(f"Registering client {client_id}")
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
    return client_id


with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(register_one, range(NUM_CLIENTS)))

print("All clients registered:", results)