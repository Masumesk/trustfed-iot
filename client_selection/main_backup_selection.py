import random
import math

from client_selection.cluster_client_share import calculate_cluster_client_share
from client_selection.selection_score import calculate_selection_scores


def main_and_backup_client_selection(
    clusters,
    client_infos,
    trust_scores, #Ti
    cluster_sample_counts,
    cluster_data_shares, #qi
    M, #main clients number
    alpha, #trust score weight in selection score
    backup_ratio, #ρ
    random_ratio
):

    
    random.seed(2) #بعدا طبق تعداد دور آموزش درست کنیم**

    m_ks = calculate_cluster_client_share(clusters, cluster_data_shares, M) #number of main clients in each cluster

    S_m = {} #main_clients
    S_b = {} #backup_clients

   

    for cluster_id, client_ids in clusters.items():

        m_k = m_ks[cluster_id]


        selection_scores = calculate_selection_scores(client_ids, #Si
            client_infos,
            trust_scores,
            cluster_sample_counts[cluster_id],
            alpha
        )


        sorted_clients = sorted(client_ids, key=selection_scores.get, reverse=True)

        random_selected = random.sample(client_ids, int(m_k*random_ratio))
        selected = list(random_selected)

        if ( m_k - len(selected) > 0 ):

            for client_id in sorted_clients:

                if client_id not in selected:

                    selected.append(client_id)

                    if len(selected) == m_k:
                        break

        S_m[cluster_id] = selected

        

        remaining_clients = [
            client_id
            for client_id in sorted_clients
            if client_id not in selected
        ]


        backup_count = min(
            len(remaining_clients)
            , max(1, math.ceil(backup_ratio * m_k))
        )

        if(backup_count > 0):
            S_b[cluster_id] = []

            for i in range(backup_count):
                S_b[cluster_id].append(remaining_clients[i])

    return S_m, S_b