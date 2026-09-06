import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from attacks.label_flip import label_flip_attack
from clustering.distribution import get_client_distribution


class Client:

    def __init__(
        self,
        client_id,
        dataset,
        indices,
        num_classes=10,
        malicious=False,
        attack_type=None,
        compute_distribution=True,
    ):

        self.client_id = client_id
        self.dataset = dataset
        self.indices = list(indices)

        self.num_samples = len(self.indices)

        if compute_distribution:

            self.distribution = get_client_distribution(
                self.dataset,
                self.indices,
                self.num_samples,
                num_classes,
            )

        else:

            self.distribution = None

        self.malicious = malicious
        self.attack_type = attack_type
        self.num_classes = num_classes

        self._loader_cache = {}

    def get_client_distribution(self):

        return {
            "client_id": self.client_id,
            "distribution": self.distribution.tolist(),
            "num_samples": self.num_samples,
        }

    def get_subset(self):

        return Subset(
            self.dataset,
            self.indices,
        )

    def get_train_loader(self, batch_size):

        batch_size = int(batch_size)

        if batch_size not in self._loader_cache:

            client_dataset = self.get_subset()
            self._loader_cache[batch_size] = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(torch.cuda.is_available()),
            )

        return self._loader_cache[batch_size]

    def local_train(
        self,
        epochs,
        batch_size,
        lr,
        reusable_model,
        global_state_dict,
        track_loss=True,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_model = reusable_model
        local_model.load_state_dict(global_state_dict)
        client_loader = self.get_train_loader(batch_size)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):

            for (images, labels) in client_loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                if self.malicious and self.attack_type == "label_flip":

                    labels = label_flip_attack(
                        labels,
                        self.num_classes,
                    )

                optimizer.zero_grad(set_to_none=True)
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if track_loss:
                    total_loss += loss.item()
                    num_batches += 1

        average_loss = (total_loss / num_batches if track_loss and num_batches > 0 else None)

        return (local_model, average_loss)

    def __repr__(self):

        return f"Client(" f"id={self.client_id}, " f"samples={self.num_samples}" f")"
