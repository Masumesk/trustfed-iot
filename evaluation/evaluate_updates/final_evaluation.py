from config import (
    MIN_REFERENCE_CLIENTS,
)

from evaluation.evaluate_updates.trust_score import (
    compute_median_update,
    compute_A_i,
    update_trust_score,
    compute_reference_scale,
)

from evaluation.evaluate_updates.evaluate_clients import (
    evaluate_clients,
)

from evaluation.evaluate_updates.backup_replacement import (
    replace_backup_clients,
)


def _build_reference_pools(
    clusters,
    main_clients,
    backup_clients,
    main_updates,
    backup_updates,
):

    cluster_updates = {}


    for cluster_id in clusters:

        updates = []


        for client_id in (
            main_clients.get(
                cluster_id,
                []
            )
        ):

            if client_id in main_updates:

                updates.append(
                    main_updates[
                        client_id
                    ]
                )


        if (
            len(updates)
            <
            MIN_REFERENCE_CLIENTS
        ):

            for client_id in (
                backup_clients.get(
                    cluster_id,
                    []
                )
            ):

                if (
                    client_id
                    in backup_updates
                ):

                    updates.append(
                        backup_updates[
                            client_id
                        ]
                    )


                if (
                    len(updates)
                    >=
                    MIN_REFERENCE_CLIENTS
                ):
                    break


        cluster_updates[
            cluster_id
        ] = updates


    return cluster_updates



def evaluate_main_clients_once(
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
    client_to_index,
):

    cluster_updates = (
        _build_reference_pools(
            clusters,
            main_clients,
            backup_clients,
            main_updates,
            backup_updates,
        )
    )


    median_cache = {}
    reference_scale_cache = {}

    round_references = {}

    evaluated_clients = set()


    for cluster_id in clusters:

        (
            reference,
            reference_cluster,
        ) = compute_median_update(

            cluster_id,
            cluster_updates,

            medoids,
            distance_matrix,

            t_near,
            client_to_index,

            median_cache,
        )


        print(
            f"Cluster {cluster_id} | "
            f"reference pool size="
            f"{len(cluster_updates.get(cluster_id, []))}"
        )


        if reference is None:

            round_references[
                cluster_id
            ] = {
                "reference": None,
                "reference_cluster":
                    None,
                "median_distance":
                    None,
            }


            print(
                f"Cluster {cluster_id} | "
                "reference unavailable | "
                "historical trust preserved"
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
                reference_updates,
            )


        median_distance = (
            reference_scale_cache[
                reference_cluster
            ]
        )


        round_references[
            cluster_id
        ] = {
            "reference":
                reference,

            "reference_cluster":
                reference_cluster,

            "median_distance":
                median_distance,
        }


        for client_id in (
            main_clients.get(
                cluster_id,
                []
            )
        ):

            if (
                client_id
                not in main_updates
            ):
                continue


            old_trust = (
                trust_scores.get(
                    client_id,
                    0.5,
                )
            )


            update = main_updates[
                client_id
            ]


            A_i = compute_A_i(
                update,
                reference,
                median_distance,
            )


            new_trust = (
                update_trust_score(
                    old_trust,
                    A_i,
                    lambda_trust,
                )
            )


            trust_scores[
                client_id
            ] = new_trust


            evaluated_clients.add(
                client_id
            )


            print(
                f"MAIN {client_id} | "
                f"A_i={A_i:.4f} | "
                f"trust "
                f"{old_trust:.4f}"
                " -> "
                f"{new_trust:.4f}"
            )


    available_main_update_clients = set(
    main_updates.keys()
)


    (
        valid_clients,
        suspicious_clients,
        unresolved_clients,
    ) = evaluate_clients(

        clusters,
        main_clients,

        trust_scores,
        trust_threshold,

        available_main_update_clients,
    )


    return (
        trust_scores,
        valid_clients,
        suspicious_clients,
        unresolved_clients,
        round_references,
        evaluated_clients,
    )



def evaluate_backups_and_replace_once(
    clusters,
    valid_clients,
    suspicious_clients,
    unresolved_clients,
    backup_clients,
    backup_updates,
    trust_scores,
    trust_threshold,
    round_references,
    client_infos,
    cluster_sample_counts,
    alpha,
    lambda_trust,
):

    evaluated_backups = set()


    for cluster_id in clusters:

        if not suspicious_clients.get(
            cluster_id,
            [],
        ):
            continue


        reference_info = (
            round_references.get(
                cluster_id
            )
        )


        if (
            not reference_info
            or
            reference_info[
                "reference"
            ] is None
        ):


            for client_id in (
                backup_clients.get(
                    cluster_id,
                    [],
                )
            ):

                if (
                    client_id
                    not in backup_updates
                ):
                    continue


                evaluated_backups.add(
                    client_id
                )


            continue


        reference = (
            reference_info[
                "reference"
            ]
        )

        median_distance = (
            reference_info[
                "median_distance"
            ]
        )


        for client_id in (
            backup_clients.get(
                cluster_id,
                []
            )
        ):

            if (
                client_id
                not in backup_updates
            ):
                continue


            old_trust = (
                trust_scores.get(
                    client_id,
                    0.5,
                )
            )


            update = (
                backup_updates[
                    client_id
                ]
            )


            A_i = compute_A_i(
                update,
                reference,
                median_distance,
            )


            new_trust = (
                update_trust_score(
                    old_trust,
                    A_i,
                    lambda_trust,
                )
            )


            trust_scores[
                client_id
            ] = new_trust


            evaluated_backups.add(
                client_id
            )


            print(
                f"BACKUP {client_id} | "
                f"A_i={A_i:.4f} | "
                f"trust "
                f"{old_trust:.4f}"
                " -> "
                f"{new_trust:.4f}"
            )


    accepted_clients = (
        replace_backup_clients(

            clusters,

            valid_clients,
            suspicious_clients,

            backup_clients,

            trust_scores,

            client_infos,
            cluster_sample_counts,

            trust_threshold,
            alpha,

            evaluated_backups,
        )
    )



    return (
        trust_scores,
        accepted_clients,
        evaluated_backups,
    )