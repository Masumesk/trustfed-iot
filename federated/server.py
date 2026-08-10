from clustering.client_clustering import client_clustering

class Server:
    def __init__(self):

        self.client_infos = {}

        self.trust_scores = {}

        #clustering
        self.cluster_assignments = {}
        self.distance_matrix = None
        self.medoids = {}
        self.cluster_samples_counts = {}
        self.cluster_data_shares = {}

        #client_selection
        self.selection_scores = {}
        self.client_roles = {}
        self.client_status = {}

    
    def receive_client_distributions(self, client_infos):
        
        self.client_infos = client_infos 

        self.trust_scores = {
            client_id: 1.0
            for client_id in client_infos
        }

        
   
    def client_clustering(
        self,
        min_samples=3,
        xi=0.05,
        min_cluster_size=None,
        assignment_threshold=0.7
    ):

        (   self.cluster_assignments,
            self.medoids,
            self.cluster_samples_counts,
            self.cluster_data_shares
        ) = client_clustering(self.client_infos, min_samples, xi, min_cluster_size, assignment_threshold)

        return self.cluster_assignments