import numpy as np
import torch

from config import MULTI_KRUM_F
from evaluation.evaluate_updates.model_update import apply_model_update
from experiments.common_server import create_baseline_app
from experiments.multi_Krum.multi_Krum import multi_krum


def aggregate_multikrum(
    *,
    global_model,
    clients,
    selected_clients,
    updates,
    current_round,
):

    aggregated_update, krum_selected, scores = multi_krum(
        updates,
        f=MULTI_KRUM_F,
    )

    with torch.no_grad():

        model_norm = torch.sqrt(
            sum(
                torch.sum(parameter.detach() ** 2)
                for parameter in global_model.parameters()
            )
        ).item()

    relative_change = np.linalg.norm(aggregated_update) / (model_norm + 1e-12)

    apply_model_update(
        global_model,
        aggregated_update,
    )

    return (
        {
            "round": current_round,
            "selected_by_krum": krum_selected,
            "scores": {
                int(client_id): float(score) for client_id, score in scores.items()
            },
            "relative_change": float(relative_change),
        },
        True,
    )


app = create_baseline_app(
    server_name="Multi-Krum",
    aggregate_round=aggregate_multikrum,
    copy_received_update=False,
)
