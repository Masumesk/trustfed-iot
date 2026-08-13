def compute_client_weights(
        accepted_clients,
        trust_scores,
        client_infos
):
    weights= {}

    for cluster_id,clients in accepted_clients.items():
        weights[cluster_id]={}

        if not clients:
            continue

        total_samples = sum(
            client_infos[client_id]["num_samples"]
            for client_id, penalty in clients
        )

        raw_weights = {}

        for client_id, penalty in clients:
            n_i = client_infos[client_id]["num_samples"]

            data_weight = n_i / total_samples
            trust_weight = trust_scores[client_id]

            raw_weight = data_weight * trust_weight


            raw_weights[client_id] = raw_weight

        total_raw_weight  = sum(raw_weights.values())

        for client_id, raw_weight in raw_weights.items():
            weights[cluster_id][client_id] = (
                    raw_weight / total_raw_weight
            )

    return weights