def calculate_selection_scores(client_ids, client_infos, trust_scores, cluster_sample_count, alpha):

    selection_scores = {}

    for client_id in client_ids:

        num_samples = client_infos[client_id]["num_samples"]

        data_share = (num_samples/cluster_sample_count)

        score_i = (
            alpha * trust_scores[client_id]
            + (1 - alpha) * data_share
        )

        selection_scores[client_id] = score_i

    return selection_scores