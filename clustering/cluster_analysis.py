
def calculate_cluster_samples_counts_size(G, client_infos):

    cluster_samples_counts = {}

    for cluster_id, client_ids in G.items():

        total_samples = 0

        for client_id in client_ids:
            total_samples += client_infos[client_id]["num_samples"]

        cluster_samples_counts[cluster_id] = total_samples

    return cluster_samples_counts


def calculate_cluster_samples_share(cluster_samples_counts):

    total_data = sum(cluster_samples_counts.values())

    cluster_share = {}

    for cluster_id, samples_count in cluster_samples_counts.items():
        cluster_share[cluster_id] = (samples_count/total_data)

    return cluster_share

    