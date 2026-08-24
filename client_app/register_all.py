import subprocess
import sys

NUM_CLIENTS=30
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
            str(client_id)
        ],
        check=True
    )

print("All clients registered")

