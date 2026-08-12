import torch

from federated.fedavg import fedavg
from models.model import MNISTCNN
from data.partition import partition_dirichlet

from data import load_mnist
from federated.client import Client
from federated.server import Server
from torch.utils.data import DataLoader


NUM_ROUNDS = 10
NUM_CLIENTS = 20

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01


#Load MNIST
train_dataset, test_dataset = load_mnist()

#Split data between clients
client_indices=partition_dirichlet(
    dataset=train_dataset,num_clients=NUM_CLIENTS)

#Create clients
clients = []

for client_id, indices in enumerate(client_indices):

    client = Client(
        client_id=client_id,
        dataset=train_dataset,
        indices=indices,
        num_classes=10
    )

    clients.append(client)
    # print(
    #     f"Client {client_id}: "
    #     f"{len(indices)} samples"
    # )

#Create server
server = Server(clients)

#Create one global model
global_model=MNISTCNN()

# Create test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

for round_idx in range (NUM_ROUNDS):
    print(f"\nRound {round_idx+1}")

    #local training
    local_models = []
    local_losses = []
    sample_counts = []

    for client in clients:

        local_model, loss = client.local_train(
            model=global_model,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )

        local_models.append(local_model)
        local_losses.append(loss)
        sample_counts.append(
            client.num_samples
        )

        print(
            f"Client {client.client_id} loss: {loss:.4f}"
        )

    #Federated Averaging
    global_state = fedavg(local_models,sample_counts)

    #Update existing global model
    global_model.load_state_dict(global_state)
    # print("Global model created successfully.")

    # average_client_loss = (
    #         sum(local_losses) / len(local_losses)
    # )
    #
    # print(
    #     f"Average local loss: "
    #     f"{average_client_loss:.4f}"
    # )

    #Evaluate global model
    global_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = global_model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total

    avg_loss = (
            sum(local_losses)
            / len(local_losses)
    )

    print(
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )
