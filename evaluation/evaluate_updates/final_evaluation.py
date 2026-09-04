from config import MIN_REFERENCE_CLIENTS

from evaluation.evaluate_updates.trust_score import (
    compute_median_update,
    compute_A_i,
    update_trust_score,
    compute_reference_scale
)

from evaluation.evaluate_updates.evaluate_clients import (
    evaluate_clients,
)

from evaluation.evaluate_updates.backup_replacement import (
    replace_backup_clients,
)


def trust_evaluation_and_backup_replacement(
    clusters,
    main_clients,
    backup_clients,
    main_updates,
    backup_updates,
    trust_scores,
    medoids,
    distance_matrix,
    t_near,
    lambda_trust,
    trust_threshold,
    alpha,
    client_infos,
    cluster_sample_counts,
    client_to_index
):


    cluster_updates = {}
    backup_requirements = {}
    median_cache = {}
    reference_scale_cache = {}

    for cluster_id in clusters:

        updates = []

        # Main updates
        for client_id in main_clients.get(
            cluster_id,
            []
        ):

            if client_id in main_updates:
                updates.append(
                    main_updates[client_id]
                )


        # Use already-available backup updates
        if len(updates) < MIN_REFERENCE_CLIENTS:

            for client_id in backup_clients.get(
                    cluster_id,
                    []
            ):

                if client_id in backup_updates:
                    updates.append(
                        backup_updates[client_id]
                    )

                if len(updates) >= MIN_REFERENCE_CLIENTS:
                    break

        cluster_updates[
            cluster_id
        ] = updates

        # If still not enough, request missing backups
        if len(updates) < MIN_REFERENCE_CLIENTS:

            needed = [
                cid
                for cid
                in backup_clients.get(
                    cluster_id, []
                )
                if cid not in backup_updates
            ]

            if needed:
                backup_requirements[
                    cluster_id
                ] = needed


    evaluated_clients = set()

    evaluated_backups = set()


    # Evaluate each cluster

    for cluster_id in clusters:

        reference, reference_cluster = (
            compute_median_update(
                cluster_id,
                cluster_updates,
                medoids,
                distance_matrix,
                t_near,
                client_to_index,
                median_cache
            )
        )

        

        print(
            f"Cluster {cluster_id} | "
            f"reference pool size="
            f"{len(cluster_updates.get(cluster_id, []))}"
        )


        # No valid reference

        if reference is None:

            print(
                f"Cluster {cluster_id} | "
                f"reference unavailable | "
                f"historical trust preserved"
            )

            continue



        reference_updates = (
            cluster_updates[
                reference_cluster
            ]
        )

        if (
            reference_cluster
            not in reference_scale_cache
        ):
            reference_scale_cache[
                reference_cluster
            ] = compute_reference_scale(
                reference,
                reference_updates
            )

        median_distance = (
            reference_scale_cache[
                reference_cluster
            ]
        )


        if reference_cluster == cluster_id:

            reference_source = (
                f"own_cluster_{cluster_id}"
            )

        else:

            reference_source = (
                f"nearest_cluster_"
                f"{reference_cluster}"
            )


        print(
            f"Cluster {cluster_id} | "
            f"reference source="
            f"{reference_source} | "
            f"reference size="
            f"{len(reference_updates)}"
        )


        # Evaluate MAIN clients

        for client_id in main_clients.get(
            cluster_id,
            []
        ):

            if client_id not in main_updates:
                continue


            old_trust = trust_scores.get(
                client_id,
                0.5
            )

            update = main_updates[
                client_id
            ]


            A_i = compute_A_i(
                update,
                reference,
                median_distance
            )


            new_trust = update_trust_score(
                old_trust,
                A_i,
                lambda_trust
            )



            trust_scores[
                client_id
            ] = new_trust


            evaluated_clients.add(
                client_id
            )


            print(
                f"Client {client_id} | "
                f"A_i={A_i:.4f} | "
                f"trust "
                f"{old_trust:.4f}"
                f" -> "
                f"{new_trust:.4f}"
            )


        # Evaluate BACKUP clients

        for client_id in backup_clients.get(
            cluster_id,
            []
        ):

            if client_id not in backup_updates:
                continue


            old_trust = trust_scores.get(
                client_id,
                0.5
            )

            update = backup_updates[
                client_id
            ]


            A_i = compute_A_i(
                update,
                reference,
                median_distance
            )


            new_trust = update_trust_score(
                old_trust,
                A_i,
                lambda_trust
            )


            trust_scores[
                client_id
            ] = new_trust


            evaluated_backups.add(
                client_id
            )


    # Split mains into valid / suspicious

    valid_clients, suspicious_clients, unresolved_clients = (
        evaluate_clients(
            clusters,
            main_clients,
            trust_scores,
            trust_threshold,
            evaluated_clients
        )
    )


    # Backup replacement

    accepted_clients = replace_backup_clients(
        clusters,
        valid_clients,
        suspicious_clients,
        backup_clients,
        trust_scores,
        client_infos,
        cluster_sample_counts,
        trust_threshold,
        alpha,
        evaluated_backups
    )

    for cluster_id in clusters:

        for client_id in unresolved_clients.get(cluster_id, []):
            accepted_clients[cluster_id].append(
                (
                    client_id,
                    float(trust_scores.get(client_id, 0.5))
                )
            )

    return (
        trust_scores,
        accepted_clients,
        backup_requirements
    )