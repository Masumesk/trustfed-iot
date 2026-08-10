import numpy as np


def hellinger_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sqrt(
        0.5 * np.sum(
            (np.sqrt(p) - np.sqrt(q)) ** 2
        )
    )

def build_distance_matrix(client_infos,client_ids):

    n = len(client_ids)

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):

            client_i = client_infos[client_ids[i]]
            client_j = client_infos[client_ids[j]]

            d = hellinger_distance(
                client_i["distribution"],
                client_j["distribution"]
            )

            matrix[i, j] = d
            matrix[j, i] = d

    return matrix