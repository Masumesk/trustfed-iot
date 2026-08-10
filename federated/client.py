from clustering.distribution import get_client_distribution
import numpy as np
from torch.utils.data import Subset


class Client:
    def __init__(
        self,
        client_id,
        dataset,
        indices,
        num_classes=10
    ):
        self.client_id = client_id

        self.dataset = dataset
        self.indices = list(indices)

        self.num_samples = len(self.indices)

        self.distribution = get_client_distribution(self.dataset, self.indices, self.num_samples, num_classes,)

    def get_client_distribution(self):
        return {
            "client_id": self.client_id,
            "distribution": self.distribution, #pi
            "num_samples": self.num_samples    #Ni
        }

    

    def get_subset(self):
        return Subset(
            self.dataset,
            self.indices
        )

    def __repr__(self):
        return (
            f"Client("
            f"id={self.client_id}, "
            f"samples={self.num_samples}"
            f")"
        )