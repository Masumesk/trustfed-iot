import numpy as np

from config import MIN_REFERENCE_CLIENTS


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

    # trim_count = int(
    #     np.floor(trim_ratio * n_clients)
    # )

    # Cannot trim if too few clients check!!
    # if 2 * trim_count >= n_clients:
    #     trim_count = 0

    trim_count = int(
        np.floor(trim_ratio * n_clients)
    )

    # if n_clients >= 3:
    #     trim_count = max(1, trim_count)

    trim_count = min(
        trim_count,
        (n_clients - 1) // 2
    )


    if trim_count == 0:

        weights = np.asarray(
            [
                client_weights[cid]
                for cid in client_ids
            ],
            dtype=updates.dtype,
        )

        weights = (
            weights / weights.sum()
        )

        return np.sum(
            updates * weights[:, None],
            axis=0,
        )



    weights_all = np.asarray(
    [
        client_weights[cid]
        for cid in client_ids
    ]
    )
    
    result = np.zeros(n_params)

    chunk_size = 65536

    for start in range(
        0,
        n_params,
        chunk_size
    ):
        end = min(
            start + chunk_size,
            n_params
        )

        
        values = updates[:, start:end]

        order = np.argsort(
            values,
            axis=0
        )

        kept_indices = order[
            trim_count:
            n_clients - trim_count,
            :
        ]

        kept_values = np.take_along_axis(
            values,
            kept_indices,
            axis=0
        )

        kept_weights = weights_all[
            kept_indices
        ]

        kept_weights = (
            kept_weights
            /
            kept_weights.sum(
                axis=0,
                keepdims=True
            )
        )

        result[start:end] = np.sum(
            kept_values * kept_weights,
            axis=0
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

        if len(updates) < MIN_REFERENCE_CLIENTS:
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