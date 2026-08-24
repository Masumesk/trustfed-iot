#server
from clustering.hellinger import build_distance_matrix

from clustering.optics import (
    optic_clustering,
    create_clusters,
    calculate_medoids,
    assign_noise
)

from clustering.cluster_analysis import (
    calculate_cluster_samples_counts_size,
    calculate_cluster_samples_share
)


def client_clustering(
    client_infos,
    min_samples,
    xi,
    min_cluster_size,
    assignment_threshold
):

    client_ids = list(client_infos.keys())
    client_to_index = { #index
        cid: idx
        for idx, cid in enumerate(client_ids)
    }

    #compute Hellinger distance matrix
    distance_matrix = build_distance_matrix(client_infos, client_ids)

    #OPTICS
    labels = optic_clustering(
        distance_matrix,
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size
    )

    #create clusters and identify noise clients
    G, Q = create_clusters(labels, client_ids)

    #calculate cluster medoids
    medoids = calculate_medoids(
        G,
        distance_matrix,
        client_to_index
    )

    #assign noise clients
    G_final = assign_noise(
        G,
        Q,
        medoids,
        distance_matrix,
        assignment_threshold=assignment_threshold,
        client_to_index=client_to_index
    )

    #recalculate medoids for final clusters
    medoids = calculate_medoids(
        G_final,
        distance_matrix,
        client_to_index
    )

    #calculate cluster sample counts
    cluster_samples_counts = (
        calculate_cluster_samples_counts_size(
            G_final,
            client_infos
        )
    )

    #calculate cluster data shares
    cluster_data_shares = (
        calculate_cluster_samples_share(
            cluster_samples_counts
        )
    )

    return (
        distance_matrix,
        G_final,
        medoids,
        cluster_samples_counts,
        cluster_data_shares
    )