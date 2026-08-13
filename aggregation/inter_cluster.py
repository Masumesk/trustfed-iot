import numpy as np


def inter_cluster_aggregation(
    cluster_updates,
    cluster_data_shares
):
    if not cluster_updates:
        raise ValueError(
            "No cluster updates available."
        )

    global_update = np.zeros_like(
        next(iter(cluster_updates.values()))
    )

    active_clusters = list(
        cluster_updates.keys()
    )

    total_share = sum(
        cluster_data_shares[k]
        for k in active_clusters
    )

    for cluster_id in active_clusters:

        cluster_weight = (
            cluster_data_shares[cluster_id]
            / total_share
        )

        global_update += (
            cluster_weight
            * cluster_updates[cluster_id]
        )

    return global_update