from sklearn.cluster import OPTICS

def optic_clustering(D, min_samples, xi, min_cluster_size):

    optics = OPTICS(
        min_samples=min_samples,
        cluster_method="xi",  # ξ Cluster Extraction
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric="precomputed",
    )

    optics.fit(D)
    labels = optics.labels_
    return labels


def create_clusters(labels, client_ids):

    G = {}  # clusters
    Q = []  # noise

    for index, label in enumerate(labels):
        client_id = client_ids[index]
        label = int(label)
        if label == -1:
            Q.append(client_id)

        else:
            if label not in G:
                G[label] = []
            G[label].append(client_id)

    return G, Q


def calculate_medoids(
    G,
    D,
    client_to_index,
):

    medoids = {}

    for cluster_id, clients in G.items():

        if len(clients) == 1:
            medoids[cluster_id] = clients[0]
            continue

        cluster_indices = [client_to_index[client_id] for client_id in clients]

        min_dist = float("inf")

        for client_i, i in zip(clients, cluster_indices):

            total_dist = 0.0

            for j in cluster_indices:
                total_dist += D[i, j]

            if total_dist < min_dist:
                min_dist = total_dist
                medoids[cluster_id] = client_i

    return medoids


def assign_noise(G, Q, medoids, D, assignment_threshold, client_to_index):

    G_final = {cluster_id: clients.copy() for cluster_id, clients in G.items()}

    if len(G_final) > 0:
        next_cluster_id = max(G_final.keys()) + 1
    else:
        next_cluster_id = 0

    medoid_indices = { cluster_id: client_to_index[medoid] for cluster_id, medoid in medoids.items()}

    for noise_client in Q:

        cluster_k_i = None
        min_dist = float("inf")

        i = client_to_index[noise_client]

        for cluster_id, j in medoid_indices.items():

            dist = D[i, j]

            if dist <= min_dist:
                min_dist = dist
                cluster_k_i = cluster_id

        if cluster_k_i is not None and min_dist <= assignment_threshold:
            G_final[cluster_k_i].append(noise_client)

        else:
            G_final[next_cluster_id] = [noise_client]
            next_cluster_id += 1

    return G_final
