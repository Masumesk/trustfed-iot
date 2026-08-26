import numpy as np


def multi_krum(client_updates, f, m=None):
   
    client_ids = list(client_updates.keys())

    n = len(client_ids)

    if 2 * f + 2 >= n:
        raise ValueError("multi-Krum condition failed:")

    updates = np.stack([
        np.asarray(
            client_updates[cid],
            dtype=np.float32
        ).reshape(-1)
        for cid in client_ids
    ])

    num_neighbors = n-f-2

    scores = {}

    for i, client_id in enumerate(client_ids):

        distances = []

        for j in range(n):

            if i != j:
                distance = np.sum(
                    (updates[i] - updates[j]) ** 2
                )

                distances.append(distance)

        distances.sort()

        score = sum(
            distances[:num_neighbors]
        )

        scores[client_id] = float(score)

    max_m = n - f - 2

    if m is None:
        m = max_m

    if m < 1 or m > max_m:
        raise ValueError(
            f"Invalid m={m}. "
            f"Must satisfy 1 <= m <= {max_m}"
        )

    # score_values = np.array([scores[cid] for cid in client_ids])

    # selected_indices = np.argsort(score_values)[:m]

    selected_clients = sorted(
        client_ids,
        key=lambda cid: (scores[cid], cid)
    )[:m]

    selected_updates = np.stack([
        np.asarray(
            client_updates[cid],
            dtype=np.float32
        ).reshape(-1)
        for cid in selected_clients
    ])

    global_update = np.mean(
        selected_updates,
        axis=0
    )

    return (
        global_update,
        selected_clients,
        scores
    )