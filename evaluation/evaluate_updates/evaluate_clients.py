def evaluate_clients(
        clusters,
        main_clients,
        trust_scores,
        trust_threshold,
        evaluated_clients
):
    valid_clients = {}  # V_Gt
    suspicious_clients = {}  # U_Gt
    unresolved_clients = {}

    for cluster_id in clusters:

        valid_clients[cluster_id] = []
        suspicious_clients[cluster_id] = []
        unresolved_clients[cluster_id] = []

        for client_id in main_clients.get(cluster_id, []):

            if client_id not in evaluated_clients:
                unresolved_clients[cluster_id].append(client_id)
                continue

            if (trust_scores.get(client_id) >= trust_threshold.get(cluster_id, 0.5)):
                valid_clients[cluster_id].append(client_id)
            else:
                suspicious_clients[cluster_id].append(client_id)

    return valid_clients, suspicious_clients, unresolved_clients