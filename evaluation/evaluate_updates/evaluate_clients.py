
def evaluate_clients(
    clusters,
    main_clients,
    trust_scores,
    trust_threshold,
    available_update_clients,
):

    valid_clients = {}
    suspicious_clients = {}

    unresolved_clients = {}


    for cluster_id in clusters:

        valid_clients[cluster_id] = []
        suspicious_clients[cluster_id] = []
        unresolved_clients[cluster_id] = []


        threshold = trust_threshold.get(
            cluster_id,
            0.5,
        )


        for client_id in main_clients.get(
            cluster_id,
            [],
        ):

            if (
                client_id
                not in available_update_clients
            ):

                unresolved_clients[
                    cluster_id
                ].append(
                    client_id
                )

                continue


            trust = trust_scores[
                client_id
            ]


            if trust >= threshold:

                valid_clients[
                    cluster_id
                ].append(
                    client_id
                )

            else:

                suspicious_clients[
                    cluster_id
                ].append(
                    client_id
                )


    return (
        valid_clients,
        suspicious_clients,
        unresolved_clients,
    )