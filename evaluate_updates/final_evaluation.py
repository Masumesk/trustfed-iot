from evaluate_updates.compute_cluster_median import compute_median_update
from evaluate_updates.normalized_distance import compute_A_i
from evaluate_updates.update_trust_score import update_trust_score

def evaluate_trust(
    clusters,
    main_clients,
    backup_clients,
    main_updates,
    backup_updates,
    trust_scores,
    medoids,
    distance_matrix,
    t_near,
    lambda_trust
):

    cluster_updates = {}

    for cluster_id in clusters:

        updates = []

        for client_id in main_clients.get(cluster_id, []):
            updates.append(main_updates[client_id])

        if len(updates) < 2:
            for client_id in backup_clients.get(cluster_id, []):
                updates.append(backup_updates[client_id])

        cluster_updates[cluster_id] = updates



    for cluster_id in clusters:

        md, nearest_cluster = compute_median_update(
            cluster_id,
            cluster_updates,
            medoids,
            distance_matrix,
            t_near
        )

        if nearest_cluster is not None:
            
            nearest_updates = cluster_updates[nearest_cluster]

            for client_id in main_clients.get(cluster_id, []):

                old_trust = trust_scores.get(client_id)

                update = main_updates[client_id]

                A_i = compute_A_i(
                    update,
                    md,
                    nearest_updates
                )

                new_trust = update_trust_score(
                    old_trust,
                    A_i,
                    lambda_trust
                )

                trust_scores[client_id] = float(round(new_trust,2))


        
            for client_id in backup_clients.get(cluster_id, []):


                old_trust = trust_scores.get(client_id)

                update = backup_updates[client_id]

                A_i = compute_A_i(
                    update,
                    md,
                    nearest_updates
                )

                new_trust = update_trust_score(
                    old_trust,
                    A_i,
                    lambda_trust
                )

                trust_scores[client_id] = float(round(new_trust,2))


    return trust_scores