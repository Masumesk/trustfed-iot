import numpy as np
import torch

from evaluation.evaluate_updates.model_update import apply_model_update
from experiments.common_server import create_baseline_app
from experiments.fedAvg.fedavg import fedavg_updates


def aggregate_fedavg(
    *,
    global_model,
    clients,
    selected_clients,
    updates,
    current_round,
):

    round_updates = [updates[client_id] for client_id in selected_clients]

    sample_counts = [clients[client_id].num_samples for client_id in selected_clients]

    aggregated_update = fedavg_updates(
        round_updates,
        sample_counts,
    )

    if not np.isfinite(aggregated_update).all():

        bad_count = int(
            np.size(aggregated_update)
            - np.count_nonzero(np.isfinite(aggregated_update))
        )

        print(
            f"[WARNING] Round "
            f"{current_round} | "
            "aggregated update contains "
            f"{bad_count} non-finite values.",
            flush=True,
        )

        return (
            {
                "round": current_round,
                "relative_change": 1e12,
                "diverged": True,
                "reason": "non_finite_aggregated_update",
            },
            False,
        )

    with torch.no_grad():

        model_sq_sum = sum(
            torch.sum(parameter.detach().double() ** 2).item()
            for parameter in global_model.parameters()
        )

    model_norm = float(np.sqrt(model_sq_sum))

    update_norm = float(
        np.linalg.norm(
            np.asarray(
                aggregated_update,
                dtype=np.float64,
            )
        )
    )

    relative_change = update_norm / (model_norm + 1e-12)

    if not np.isfinite(relative_change):

        print(
            f"[WARNING] Round "
            f"{current_round} | "
            "relative_change became "
            f"{relative_change}.",
            flush=True,
        )

        return (
            {
                "round": current_round,
                "relative_change": 1e12,
                "diverged": True,
                "reason": "non_finite_relative_change",
            },
            False,
        )

    apply_model_update(
        global_model,
        aggregated_update,
    )

    return (
        {
            "round": current_round,
            "relative_change": float(relative_change),
        },
        True,
    )


app = create_baseline_app(
    server_name="FedAvg",
    aggregate_round=aggregate_fedavg,
    copy_received_update=True,
)
