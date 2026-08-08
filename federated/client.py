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

        self.histogram = self._compute_histogram(num_classes)
        self.distribution = self._normalize_histogram()

    def _compute_histogram(self, num_classes):
        targets = np.array(self.dataset.targets)

        labels = targets[self.indices]

        return np.bincount(
            labels,
            minlength=num_classes
        )

    def _normalize_histogram(self):
        return self.histogram / self.num_samples

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