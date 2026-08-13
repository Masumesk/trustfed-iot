import numpy as np


def get_client_update(
    client_id,
    main_updates,
    backup_updates
):
    if client_id in main_updates:
        return main_updates[client_id]

    if client_id in backup_updates:
        return backup_updates[client_id]

    raise ValueError(
        f"No update found for client {client_id}"
    )


def weighted_trimmed_mean(
    client_updates,
    client_weights,
    trim_ratio=0.2
):
    client_ids = list(client_updates.keys())

    updates = np.stack(
        [client_updates[cid] for cid in client_ids]
    )

    n_clients, n_params = updates.shape

    trim_count = int(
        np.floor(trim_ratio * n_clients)
    )

    # Cannot trim if too few clients
    if 2 * trim_count >= n_clients:
        trim_count = 0

    result = np.zeros(n_params)

    for p in range(n_params):

        values = updates[:, p]

        order = np.argsort(values)

        if trim_count > 0:
            kept_indices = order[
                trim_count:n_clients-trim_count
            ]
        else:
            kept_indices = order

        kept_ids = [
            client_ids[idx]
            for idx in kept_indices
        ]

        kept_values = values[kept_indices]

        weights = np.array([
            client_weights[cid]
            for cid in kept_ids
        ])

        weights = weights / weights.sum()

        result[p] = np.sum(
            kept_values * weights
        )

    return result

def aggregate_clusters(
    clusters,
    accepted_clients,
    main_updates,
    backup_updates,
    client_weights,
    trim_ratio=0.2
):
    cluster_updates = {}

    for cluster_id in clusters:

        accepted = accepted_clients.get(
            cluster_id,
            []
        )

        updates = {}

        for client_id, penalty in accepted:

            if client_id in main_updates:
                updates[client_id] = (
                    main_updates[client_id]
                )

            elif client_id in backup_updates:
                updates[client_id] = (
                    backup_updates[client_id]
                )

        if len(updates) == 0:
            continue

        if len(updates) < 3:
            # weighted mean fallback
            result = np.zeros_like(
                next(iter(updates.values()))
            )

            for client_id, update in updates.items():
                result += (
                    client_weights[cluster_id][client_id]
                    * update
                )

        else:
            result = weighted_trimmed_mean(
                updates,
                client_weights[cluster_id],
                trim_ratio
            )

        cluster_updates[cluster_id] = result

    return cluster_updates