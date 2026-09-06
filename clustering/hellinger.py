import numpy as np

def hellinger_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def build_distance_matrix(client_infos, client_ids):

    n = len(client_ids)
    matrix = np.zeros( (n, n), dtype=float,)

    sqrt_distributions = {
        client_id: np.sqrt(
            np.asarray(
                client_infos[client_id]["distribution"],
                dtype=float,
            )
        )
        for client_id in client_ids
    }

    for i in range(n):

        sqrt_i = sqrt_distributions[client_ids[i]]

        for j in range(i + 1, n):

            sqrt_j = sqrt_distributions[client_ids[j]]
            d = np.sqrt(0.5 * np.sum((sqrt_i - sqrt_j) ** 2))
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix
