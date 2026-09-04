import numpy as np


def multi_krum(client_updates, f, m=None):

    client_ids = list(client_updates.keys())

    n = len(client_ids)

    if 2 * f + 2 >= n:
        raise ValueError("multi-Krum condition failed:")

    updates = np.stack(
        [
            np.asarray(client_updates[cid], dtype=np.float32).reshape(-1)
            for cid in client_ids
        ]
    )

    num_neighbors = n - f - 2

    # هر فاصله فقط یک بار محاسبه می‌شود
    distances_per_client = [[] for _ in range(n)]

    for i in range(n):

        for j in range(i + 1, n):

            distance = np.sum((updates[i] - updates[j]) ** 2)

            distances_per_client[i].append(distance)

            distances_per_client[j].append(distance)

    scores = {}

    for i, client_id in enumerate(client_ids):

        distances = distances_per_client[i]

        distances.sort()

        score = sum(distances[:num_neighbors])

        scores[client_id] = float(score)

    max_m = n - f - 2

    if m is None:
        m = max_m

    if m < 1 or m > max_m:
        raise ValueError(f"Invalid m={m}. " f"Must satisfy " f"1 <= m <= {max_m}")

    selected_clients = sorted(client_ids, key=lambda cid: (scores[cid], cid))[:m]

    # از updates موجود استفاده می‌کنیم؛
    # دوباره stack نمی‌کنیم.
    client_index = {cid: index for index, cid in enumerate(client_ids)}

    selected_indices = [client_index[cid] for cid in selected_clients]

    selected_updates = updates[selected_indices]

    global_update = np.mean(selected_updates, axis=0)

    return (global_update, selected_clients, scores)
