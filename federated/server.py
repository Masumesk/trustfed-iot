from clustering.client_clustering import client_clustering
from client_selection.main_backup_selection import main_and_backup_client_selection

class Server:
    def __init__(self):

        self.client_infos = {}
        self.N = 0

        self.trust_scores = {}

        #clustering
        self.clusters = {} #G
        self.distance_matrix = None #D
        self.medoids = {}
        self.cluster_samples_counts = {} #Ni
        self.cluster_data_shares = {} #qk
        self.k=0 #cluster count

        #client_selection
        self.M=0 
        self.main_clients = {}
        self.backup_clients = {}

    
    def receive_client_distributions(self, client_infos):
        
        self.client_infos = client_infos 
        self.N=len(self.client_infos)

        for client_id in client_infos:
            if client_id not in self.trust_scores:
                self.trust_scores[client_id] = 0.5 

        
   
    def client_clustering(
        self,
        min_samples=3,
        xi=0.05,
        min_cluster_size=None,
        assignment_threshold=0.7
    ):

        (   self.clusters,
            self.medoids,
            self.cluster_samples_counts,
            self.cluster_data_shares
        ) = client_clustering(self.client_infos, min_samples, xi, min_cluster_size, assignment_threshold)
        self.k=len(self.clusters)
        self.M=max(
            self.k,
            round(self.N * 0.3)
        )

        

    def main_and_backup_client_selection( #در هر دور آموزش 
        self,
        alpha=0.5,
        backup_ratio=0.5,
        random_ratio=0.5
    ):

        (
            self.main_clients,
            self.backup_clients
        ) = main_and_backup_client_selection(
            self.clusters,
            self.client_infos,
            self.trust_scores,
            self.cluster_samples_counts,
            self.cluster_data_shares,
            self.M,
            alpha,
            backup_ratio,
            random_ratio
        )

        

    


