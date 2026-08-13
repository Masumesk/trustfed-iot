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

import numpy as np


def compute_A_i(update, reference, reference_updates):

    distance = np.sqrt(np.sum((update - reference) ** 2))

    distances = []

    for ref_update in reference_updates:

        d = np.sqrt(np.sum((ref_update - reference)**2))
        distances.append(d)


    median_distance = np.median(distances)
    
    A_i = ( distance / (median_distance + (1e-8)) )

    return A_i

def update_trust_score(old_trust, A_i, lambda_trust):

    new_trust = (lambda_trust*old_trust + (1 - lambda_trust)*(1 / (1 + A_i)))
    return new_trust