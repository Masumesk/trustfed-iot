from clustering.client_clustering import client_clustering
from client_selection.main_backup_selection import main_and_backup_client_selection
from evaluate_updates.final_evaluation import trust_evaluation_and_backup_replacement
from aggregation.client_weights import compute_client_weights
from aggregation.intra_cluster import aggregate_clusters
from aggregation.inter_cluster import inter_cluster_aggregation
from federated.model_update import apply_model_update

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

        #train
        self.main_updates = {}
        self.backup_updates = {}
        self.current_round = 0

        #aggregation
        self.accepted_clients = {}
        self.client_weights = {}
        self.cluster_updates = {}
        self.global_update = None

    
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

        (   self.distance_matrix,
            self.clusters,
            self.medoids,
            self.cluster_samples_counts,
            self.cluster_data_shares
        ) = client_clustering(self.client_infos, min_samples, xi, min_cluster_size, assignment_threshold)
        self.k=len(self.clusters)
        self.M=max(
            self.k,
            round(self.N * 0.3)
        )

        

    def main_and_backup_client_selection( 
        self,
        alpha=0.5,
        backup_ratio=0.5,
        random_ratio=0.5,
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
            random_ratio,
            self.current_round
        )

    def receive_client_update(
        self,
        client_id,
        update
    ):

        for clients in self.main_clients.values():
            if client_id in clients:

                self.main_updates[client_id] = update
                return

        for clients in self.backup_clients.values():
            if client_id in clients:

                self.backup_updates[client_id] = update
                return


    def trust_evaluation_and_backup_replacement(
        self,
        t_near=0.7,
        lambda_trust=0.5,
        trust_threshold = { #به تعداد خوشه ها
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3:0.5
        },
        alpha=0.5
    ):

        self.trust_scores, self.accepted_clients = (
            trust_evaluation_and_backup_replacement(
                self.clusters,
                self.main_clients,
                self.backup_clients,
                self.main_updates,
                self.backup_updates,
                self.trust_scores,
                self.medoids,
                self.distance_matrix,
                t_near,
                lambda_trust,
                trust_threshold,
                alpha,
                self.client_infos,
                self.cluster_samples_counts
            )
        )

    def start_round(self, round_id):

        self.current_round = round_id

        self.main_clients = {}
        self.backup_clients = {}

        self.main_updates = {}
        self.backup_updates = {}

        self.accepted_clients = {}
        self.client_weights = {}
        self.cluster_updates = {}
        self.global_update = None

    def create_training_package(
        self,
        global_model,
        local_epochs,
        batch_size,
        learning_rate
    ):

        return {
            "global_model": global_model,
            "round_id": self.current_round,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

    def aggregate(
            self,
            global_model,
            trim_ratio=0.2
    ):

        self.client_weights = compute_client_weights(
            self.accepted_clients,
            self.trust_scores,
            self.client_infos
        )

        self.cluster_updates = aggregate_clusters(
            self.clusters,
            self.accepted_clients,
            self.main_updates,
            self.backup_updates,
            self.client_weights,
            trim_ratio
        )

        self.global_update = (
            inter_cluster_aggregation(
                self.cluster_updates,
                self.cluster_data_shares
            )
        )

        global_model = apply_model_update(
            global_model,
            self.global_update
        )

        return global_model

        

    


