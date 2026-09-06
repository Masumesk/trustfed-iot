from client_app.run_round import (
        run_client_round,
)

def run_client_round_task(
    client_id,
    server_url,
    send_immediately=True,
    expected_round=None,

):

    return run_client_round(
        client_id,
        server_url,
        send_immediately=send_immediately,
        expected_round=expected_round,
    )