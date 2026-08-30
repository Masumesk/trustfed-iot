import torch
import numpy as np

from config import (
    PARTICIPATION_RATIO,
    OPTICS_MIN_SAMPLES,
    OPTICS_XI,
    OPTICS_MIN_CLUSTER_SIZE,
    NOISE_ASSIGNMENT_THRESHOLD,
    SELECTION_ALPHA,
    BACKUP_RATIO,
    RANDOM_RATIO,
    T_NEAR,
    LAMBDA_TRUST,
    TRUST_THRESHOLD,
    INITIAL_TRUST,
    TRIM_RATIO,
)

from clustering.client_clustering import client_clustering
from client_selection.main_backup_selection import main_and_backup_client_selection
from evaluation.global_evaluate import evaluate_model
from evaluation.evaluate_updates.final_evaluation import trust_evaluation_and_backup_replacement
from aggregation.client_weights import compute_client_weights
from aggregation.intra_cluster import aggregate_clusters
from aggregation.inter_cluster import inter_cluster_aggregation
from evaluation.evaluate_updates.model_update import apply_model_update
from models.model import MNISTCNN


class Server:
    def __init__(self):

        self.client_infos = {}
        self.N = 0
        # self.client_ids = list(
        #     self.client_infos.keys()
        # )
        self.client_to_index = {}  # index

        self.trust_scores = {}

        # clustering
        self.clusters = {}  # G
        self.distance_matrix = None  # D
        self.medoids = {}
        self.cluster_samples_counts = {}  # Ni
        self.cluster_data_shares = {}  # qk
        self.k = 0  # cluster count

        # client_selection
        self.M = 0
        self.main_clients = {}
        self.backup_clients = {}

        # train
        self.main_updates = {}
        self.backup_updates = {}
        self.current_round = 0

        # aggregation
        self.accepted_clients = {}
        self.client_weights = {}
        self.cluster_updates = {}
        self.global_update = None
        self.model_relative_change = None
        self.backup_requirements = {}

        # global model
        self.global_model = MNISTCNN()

        # evaluation
        self.val_dataset = None
        self.test_dataset = None

    def receive_client_distributions(self, client_infos):

        self.client_infos = client_infos
        sorted_client_ids = sorted(
            self.client_infos.keys()
        )

        self.client_to_index = {
            cid: i
            for i, cid in enumerate(
                sorted_client_ids
            )
        }
        # self.client_to_index = {  # index
        #     cid: i
        #     for i, cid in enumerate(self.client_infos.keys())
        # }
        self.N = len(self.client_infos)

        for client_id in client_infos:
            if client_id not in self.trust_scores:
                self.trust_scores[client_id] = INITIAL_TRUST

    def client_clustering(
            self,
            min_samples=OPTICS_MIN_SAMPLES,
            xi=OPTICS_XI,
            min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
            assignment_threshold=NOISE_ASSIGNMENT_THRESHOLD,
    ):

        (self.distance_matrix,
         self.clusters,
         self.medoids,
         self.cluster_samples_counts,
         self.cluster_data_shares
         ) = client_clustering(self.client_infos, min_samples, xi, min_cluster_size, assignment_threshold)
        self.k = len(self.clusters)
        self.M = max(
            self.k,
            round(self.N * PARTICIPATION_RATIO)
        )

    def main_and_backup_client_selection(
            self,
            alpha=SELECTION_ALPHA,
            backup_ratio=BACKUP_RATIO,
            random_ratio=RANDOM_RATIO,
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

    def prepare_backup_request(
            self
    ):
        """
        Determine which clusters need backup updates.
        Returns dict: {cluster_id: [backup_client_ids]}.
        """
        from evaluation.evaluate_updates.trust_score import (
            MIN_REFERENCE_CLIENTS,
        )

        requirements = {}

        for cluster_id in self.clusters:
            main_count = sum(
                1
                for cid
                in self.main_clients.get(
                    cluster_id, []
                )
                if cid in self.main_updates
            )

            if main_count < MIN_REFERENCE_CLIENTS:
                backups = [
                    cid
                    for cid
                    in self.backup_clients.get(
                        cluster_id, []
                    )
                    if cid not in self.backup_updates
                ]

                if backups:
                    requirements[
                        cluster_id
                    ] = backups

        return requirements

    def request_backup_updates(
            self,
            cluster_id,
            client_id,
            update
    ):
        """
        Receive a single backup client's update on demand.
        """
        self.backup_updates[
            client_id
        ] = update

    def trust_evaluation_and_backup_replacement(
            self,
            t_near=T_NEAR,
            lambda_trust=LAMBDA_TRUST,
            trust_threshold_value=TRUST_THRESHOLD,
            alpha=SELECTION_ALPHA,
    ):

        trust_threshold = {
            cluster_id: trust_threshold_value
            for cluster_id in self.clusters.keys()
        }

        (
            self.trust_scores,
            self.accepted_clients,
            self.backup_requirements,
        ) = (
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
                self.cluster_samples_counts,
                self.client_to_index
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
        self.backup_requirements = {}

    def create_training_package(
            self,
            local_epochs,
            batch_size,
            learning_rate
    ):

        return {
            "global_model": self.global_model,
            "round_id": self.current_round,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

    def aggregate(
            self,
            trim_ratio=TRIM_RATIO,
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

        with torch.no_grad():
            previous_model_norm = torch.sqrt(
                sum(
                    torch.sum(
                        parameter.detach() ** 2
                    )
                    for parameter
                    in self.global_model.parameters()
                )
            ).item()

        update_norm = np.linalg.norm(
            self.global_update
        )

        self.model_relative_change = (
                update_norm
                / (previous_model_norm + 1e-12)
        )

        self.global_model = apply_model_update(
            self.global_model,
            self.global_update
        )

        return self.global_model

    def get_model_state(self):

        return {
            k: v.cpu().tolist()
            for k, v in self.global_model.state_dict().items()
        }

    def evaluate(self, use_val=True):
        """
        Evaluate model.
        use_val=True: use validation set (for training/early stopping)
        use_val=False: use test set (for final evaluation only)
        """
        dataset = self.val_dataset if use_val else self.test_dataset
        return evaluate_model(
            self.global_model,
            dataset
        )





