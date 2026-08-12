import copy
import numpy as np
from scipy.ndimage import histogram
from torch.utils.data import Subset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

        histogram=np.bincount(
            labels,
            minlength=num_classes
        )
        return histogram

    def _normalize_histogram(self):
        if self.num_samples == 0:
            return np.zeros_like(
                self.histogram,
                dtype=float
            )

        return self.histogram / self.num_samples

    def get_client_distribution(self):
        return {
            "client_id": self.client_id,
            "distribution": self.distribution,
            "num_samples": self.num_samples
        }

    def get_subset(self):
        return Subset(
            self.dataset,
            self.indices
        )


    def local_train(self, model, epochs=1, batch_size=32, lr=0.01):
        local_model = copy.deepcopy(model)

        client_dataset = self.get_subset()

        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=lr
        )

        local_model.train()

        total_loss = 0.0
        num_batches=0

        for epoch in range(epochs):

            for images, labels in client_loader:
                optimizer.zero_grad()

                outputs = local_model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()
                num_batches+=1

        average_loss = total_loss / num_batches

        return local_model, average_loss

    def __repr__(self):
        return (
            f"Client("
            f"id={self.client_id}, "
            f"samples={self.num_samples}"
            f")"
        )