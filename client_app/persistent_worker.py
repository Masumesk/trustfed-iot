def run_client_round_task(
    client_id,
    server_url,
):
    
    from client_app.run_round import (
        run_client_round,
    )

    return run_client_round(
        client_id,
        server_url,
    )