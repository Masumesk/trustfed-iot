def calculate_cluster_client_share(clusters, cluster_data_shares, M):
    m_ks = {}  
    selected=0

    for cluster_id, client_ids in clusters.items(): 

        m_k = max(
            1,
            min(
            round(M * cluster_data_shares[cluster_id]),
            len(client_ids)
            )
        )
        
        m_ks[cluster_id] = m_k
        selected+=m_k

    while (selected < M):

        candidates = [ 
            cluster_id
            for cluster_id, client_ids in clusters.items()
            if m_ks[cluster_id] < len(client_ids)
        ]

        if not candidates:
            break

        chosen = candidates[0]
        for k in candidates:
            if ( m_ks[k] <  m_ks[chosen]):
                chosen = k

        m_ks[chosen] += 1
        selected += 1

    while (selected > M):
    
            candidates = [ 
                cluster_id
                for cluster_id, client_ids in clusters.items()
                if m_ks[cluster_id] > 1
            ]
    
            if not candidates:
                break
    
            chosen = candidates[0]
            for k in candidates:
                if ( m_ks[k] >  m_ks[chosen]):
                    chosen = k
    
            m_ks[chosen] -= 1
            selected -= 1

    return m_ks

    