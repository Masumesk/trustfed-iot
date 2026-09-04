import numpy as np

from config import MIN_REFERENCE_CLIENTS


def get_cached_median(cluster_id, cluster_updates, median_cache):
    if cluster_id not in median_cache:

        median_cache[cluster_id] = np.median(
            np.stack(cluster_updates[cluster_id]), axis=0
        )

    return median_cache[cluster_id]


def compute_median_update(
    cluster_id,
    cluster_updates,
    medoids,
    distance_matrix,
    t_near,
    client_to_index,
    median_cache,
):

    updates = cluster_updates.get(cluster_id, [])

    if len(updates) >= MIN_REFERENCE_CLIENTS:

        reference = get_cached_median(cluster_id, cluster_updates, median_cache)

        return reference, cluster_id

    cur_md = medoids[cluster_id]

    nearest_cluster = None
    min_dist = float("inf")

    for candidate_id, candidate_updates in cluster_updates.items():

        if (
            len(candidate_updates) >= MIN_REFERENCE_CLIENTS
            and candidate_id != cluster_id
        ):

            candidate_medoid = medoids[candidate_id]

            i = client_to_index[cur_md]  # index
            j = client_to_index[candidate_medoid]

            distance = distance_matrix[i, j]

            if distance < min_dist:
                min_dist = distance
                nearest_cluster = candidate_id

    if nearest_cluster is not None and min_dist <= t_near:

        reference = get_cached_median(
            nearest_cluster,
            cluster_updates,
            median_cache
        )

    

        return reference, nearest_cluster

    return None, None


def compute_reference_scale(reference, reference_updates):
    distances = np.empty(len(reference_updates), dtype=np.float64)

    for idx, ref_update in enumerate(reference_updates):

        distances[idx] = np.sqrt(np.sum((ref_update - reference) ** 2))

    return np.median(distances)


def compute_A_i(update, reference, median_distance):

    distance = np.sqrt(np.sum((update - reference) ** 2))

    A_i = distance / (median_distance + 1e-8)

    return A_i


def update_trust_score(old_trust, A_i, lambda_trust):

    new_trust = (lambda_trust*old_trust + (1 - lambda_trust)*(1 / (1 + A_i)))
    return float(new_trust)