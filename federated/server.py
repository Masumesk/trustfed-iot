from clustering.hellinger import build_distance_matrix
from clustering.optics import (optic_clustering,create_clusters,calculate_medoids,assign_noise)
from clustering.cluster_analysis import (calculate_cluster_samples_counts_size,calculate_cluster_samples_share)

class Server:
    def __init__(self, clients):
        self.clients = clients

        # Server-side state
        self.trust_scores = {
            client.client_id: 1.0
            for client in clients
        }

        self.cluster_assignments = {}
        self.selection_scores = {}
        self.client_roles = {}
        self.client_status = {}

        self.distance_matrix = None
        self.medoids = {}
        self.cluster_samples_counts = {}
        self.cluster_data_shares = {}

        
    def client_clustering(self,min_samples=2,xi=0.02,min_cluster_size=None,assignment_threshold=0.6):

        #compute hellinger distances
        self.distance_matrix = build_distance_matrix(self.clients)
        #run optics
        labels = optic_clustering(self.distance_matrix,min_samples=min_samples,xi=xi,min_cluster_size=min_cluster_size)
        #create clusters from optics lables
        G, Q = create_clusters(labels)
        #calculate cluster medoids and assign noise clients to nearest clusters
        self.medoids = calculate_medoids(G,self.distance_matrix)
        G_final = assign_noise(G,Q,self.medoids,self.distance_matrix,assignment_threshold=assignment_threshold)
        #compute cluster samples and data shares
        self.cluster_samples_counts = (calculate_cluster_samples_counts_size(G_final,self.clients))
        self.cluster_data_shares = (calculate_cluster_samples_share(self.cluster_samples_counts))
        self.cluster_assignments = G_final
        return G_final
