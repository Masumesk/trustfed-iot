import numpy as np

def compute_median_update(cluster_id, cluster_updates, medoids, distance_matrix, t_near):
    
    updates = cluster_updates.get(cluster_id, [])

    if len(updates) >= 2:

        md = np.median(
            np.stack(updates),
            axis=0
        )

        return md, cluster_id



    cur_md = medoids[cluster_id]

    nearest_cluster = None
    min_dist = float("inf")

    for candidate_id, candidate_updates in cluster_updates.items():

        if ( len(candidate_updates) >= 2 and candidate_id != cluster_id ):

            candidate_medoid = medoids[candidate_id]

            distance = distance_matrix[cur_md,candidate_medoid]

            if distance < min_dist:
                min_dist = distance
                nearest_cluster = candidate_id


    
    if ( nearest_cluster is not None and min_dist <= t_near):

        md = np.median(
            np.stack(
                cluster_updates[nearest_cluster]
            ),
            axis=0
        )

        return md, nearest_cluster


    return None, None