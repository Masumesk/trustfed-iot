from clustering.distribution import get_client_distribution
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import copy
import torch
import torch.nn as nn

class Client:
    def __init__(
        self,
        client_id,
        dataset,
        indices,
        num_classes=10,
        malicious = False,
        attack = None
    ):
        self.client_id = client_id

        self.dataset = dataset
        self.indices = list(indices)

        self.num_samples = len(self.indices)

        self.distribution = get_client_distribution(self.dataset, self.indices, self.num_samples, num_classes,)

        self.training_package= None

        self.malicious = malicious
        self.attack = attack

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


    def local_train(self, model, epochs, batch_size, lr):
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

    def receive_training_package(self, package):
        self.training_package = package


    def train_received_package(self):

        package = self.training_package
        return self.local_train(
            model=package["global_model"],
            epochs=package["local_epochs"],
            batch_size=package["batch_size"],
            lr=package["learning_rate"]
        )

    def __repr__(self):
        return (
            f"Client("
            f"id={self.client_id}, "
            f"samples={self.num_samples}"
            f")"
        )