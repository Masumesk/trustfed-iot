from client_selection.selection_score import calculate_selection_scores
from config import SUSPICIOUS_PENALTY


def replace_backup_clients(
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

):
    accepted_clients = {}

    for cluster_id in clusters:

        accepted_clients[cluster_id] = []
        threshold = trust_threshold.get(
            cluster_id,
            0.5
        )
        for client_id in valid_clients.get(cluster_id, []):
            accepted_clients[cluster_id].append((client_id, 1.0))  # λ_pi for valid clients is 1

        backups = list(backup_clients.get(cluster_id, []))

        for suspicious_id in suspicious_clients.get(cluster_id, []):

            valid_backups = []
            for client_id in backups:
                trust = trust_scores.get(client_id, 0.5)
                if (client_id in evaluated_backups and trust >= threshold):
                    valid_backups.append(client_id)

            if valid_backups:
                s_scores = calculate_selection_scores(
                    valid_backups,
                    client_infos,
                    trust_scores,
                    cluster_sample_counts[cluster_id],
                    alpha
                )

                best_backup = valid_backups[0]
                for client_id in valid_backups:
                    if (s_scores[client_id] > s_scores[best_backup]):
                        best_backup = client_id

                accepted_clients[cluster_id].append((best_backup, 1.0))
                backups.remove(best_backup)


            else:
                available_backups = [
                    client_id

                    for client_id in backups

                    if client_id
                    in evaluated_backups
                ]

                candidates = (
                    [suspicious_id]
                    + available_backups
                )
                best_candidate = candidates[0]

                for client_id in candidates:
                    if (trust_scores.get(client_id) > trust_scores.get(best_candidate)):
                        best_candidate = client_id

                accepted_clients[cluster_id].append((best_candidate, SUSPICIOUS_PENALTY))

                if (best_candidate in backups):
                    backups.remove(best_candidate)

    return accepted_clients