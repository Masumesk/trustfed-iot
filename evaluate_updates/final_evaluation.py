from evaluate_updates.trust_score import compute_median_update, compute_A_i, update_trust_score
from evaluate_updates.evaluate_clients import evaluate_clients
from evaluate_updates.backup_replacement import replace_backup_clients
def trust_evaluation_and_backup_replacement(
    clusters,
    main_clients, #S_m
    backup_clients, #S_b
    main_updates, 
    backup_updates,
    trust_scores, 
    medoids,
    distance_matrix, #D
    t_near, #threshold for finding near clusters
    lambda_trust, #weight of old trust score in trust score
    trust_threshold, 
    alpha, #weight of trust score in selection score
    client_infos,
    cluster_sample_counts #Ni
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

        #compute median 
        md, nearest_cluster = compute_median_update(
            cluster_id,
            cluster_updates,
            medoids,
            distance_matrix,
            t_near
        )

        #for clusters with just one update evalute updates base on nearest cluster
        # ToDo:change nearest_cluster name
        if nearest_cluster is not None:
            
            nearest_updates = cluster_updates[nearest_cluster]

            for client_id in main_clients.get(cluster_id, []):

                old_trust = trust_scores.get(client_id)

                update = main_updates[client_id]

                #normalized distance to cluster median
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

                trust_scores[client_id] = round(float(new_trust),2)


            #update trust score for backups
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

                trust_scores[client_id] = round(float(new_trust),2)

    valid_clients, suspicious_clients = evaluate_clients( #V_Gt #U_Gt
        clusters,
        main_clients,
        trust_scores,
        trust_threshold
    )

    accepted_clients= replace_backup_clients(
        clusters,
        valid_clients,
        suspicious_clients,
        backup_clients,
        trust_scores,
        client_infos,
        cluster_sample_counts,
        trust_threshold,
        alpha,
    )

    return trust_scores,accepted_clients
    