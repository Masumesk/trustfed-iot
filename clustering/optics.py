from sklearn.cluster import OPTICS
import numpy as np

def optic_clustering(D,min_samples=3,xi=0.03,min_cluster_size=None):
     
    optics = OPTICS(
        min_samples=min_samples,
        cluster_method='xi', # ξ Cluster Extraction
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric="precomputed"
    )

    optics.fit(D)
    labels = optics.labels_
    return labels

def create_clusters(labels):

    G = {}  #clusters
    Q = []  #noise

    for client_id, label in enumerate(labels):
        label = int(label)
        if label == -1:
            Q.append(client_id)

        else:
            if label not in G:
                G[label] = []
            G[label].append(client_id)

    return G , Q

def calculate_medoids(G, D):

    medoids = {}

    for cluster_id, clients in G.items():
        if len(clients) == 1:
            medoids[cluster_id] = clients[0]

        else:
            min_dist = float("inf")

            for client_i in clients:
                total_dist = 0.0

                for client_j in clients:
                    total_dist += D[client_i,client_j]

                if total_dist < min_dist:
                    min_dist = total_dist
                    medoids[cluster_id] = client_i

    return medoids

def assign_noise(G,Q,medoids,D,assignment_threshold):

    G_final = {
        cluster_id: clients.copy()
        for cluster_id, clients in G.items()
    }

    if len(G_final) > 0:
        next_cluster_id = max(G_final.keys()) + 1
    else:
        next_cluster_id = 0

    for noise_client in Q:
        cluster_k_i = None #nearest cluster
        min_dist = float("inf")

        for cluster_id, medoid in medoids.items():

            dist = D[noise_client, medoid]

            if dist <= min_dist:
                min_dist = dist
                cluster_k_i = cluster_id

        if (cluster_k_i is not None and min_dist <= assignment_threshold):
            G_final[cluster_k_i].append(noise_client)

        else:
            G_final[next_cluster_id] = [noise_client]
            next_cluster_id += 1

    return G_final

            