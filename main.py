import numpy as np
import torch

from attacks.attack_manager import apply_attack
from data import load_mnist
from data.partition import partition_dirichlet
from federated.client import Client
from federated.server import Server
from federated.model_update import compute_model_update
from models.model import MNISTCNN
from torch.utils.data import random_split
from evaluate_updates.model_evaluation  import evaluate_validation

full_train_dataset, test_dataset = load_mnist()

# ToDO:Define them all Once
train_size = 55000
val_size = 5000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)



client_indices = partition_dirichlet(
    dataset=train_dataset,
    num_clients=30,
    alpha=0.3,
    min_samples=100,
    seed=42
)
# print("Train size:", len(train_dataset))
# print("Validation size:", len(val_dataset))
# print("Test size:", len(test_dataset))
#
# print(
#     "Samples assigned to clients:",
#     sum(len(indices) for indices in client_indices)
# )

ATTACK_TYPE = "gaussian"
clients = []
MALICIOUS_RATIO = 0.2
np.random.seed(42)

num_malicious = int(
    30 * MALICIOUS_RATIO
)

malicious_ids = set(
    np.random.choice(
        30,
        num_malicious,
        replace=False
    )
)
for client_id, indices in enumerate(client_indices):
    client = Client(
        client_id=client_id,
        dataset=train_dataset,
        indices=indices,
        num_classes=10,
        malicious=(client_id in malicious_ids),
        attack_type=ATTACK_TYPE
    )

    clients.append(client)

client_id_to_client = {
    client.client_id: client
    for client in clients
}

clients_by_id = {
    client.client_id: client
    for client in clients
}

for client in clients:
    print(
        client.client_id,
        client.malicious
    )

server = Server()

client_infos = {}

for client in clients:
    info = client.get_client_distribution()
    client_infos[client.client_id] = info

server.receive_client_distributions(client_infos)

server.client_clustering()

print("Clusters:")
print(server.clusters)

print("Cluster sample counts:")
print(server.cluster_samples_counts)

print("Cluster data shares:")
print(server.cluster_data_shares)


global_model = MNISTCNN()

NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODEL_CHANGE_THRESHOLD = 0.01
VAL_LOSS_CHANGE_THRESHOLD = 0.01
PATIENCE = 3


previous_val_loss = None
stable_checks = 0

for round_id in range(1, NUM_ROUNDS + 1):

    print(f"round {round_id}")

   
    server.start_round(round_id)
    server.main_and_backup_client_selection()

    print("Main clients:")
    print(server.main_clients)

    print("Backup clients:")
    print(server.backup_clients)


    training_package = server.create_training_package(
        global_model=global_model,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )


    for cluster_id, client_ids in server.main_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            client.receive_training_package(training_package)


    for cluster_id, client_ids in server.backup_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            client.receive_training_package(training_package)


    
    for cluster_id, client_ids in server.main_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            local_model, loss = (client.train_received_package())

            update = compute_model_update(
                global_model,
                local_model
            )

            if client.malicious:
                update = apply_attack(
                    update,
                    ATTACK_TYPE
                )

            server.receive_client_update(
                client_id,
                update
            )

            print(
                f"MAIN | "
                f"Round {round_id} | "
                f"Cluster {cluster_id} | "
                f"Client {client_id} | "
                f"Loss = {loss:.4f}"
            )


    for cluster_id, client_ids in server.backup_clients.items():

        for client_id in client_ids:

            client = clients_by_id[client_id]

            local_model, loss = (client.train_received_package())

            update = compute_model_update(
                global_model,
                local_model
            )

            if client.malicious:
                update = apply_attack(
                    update,
                    ATTACK_TYPE
                )

            server.receive_client_update(
                client_id,
                update
            )

            print(
                f"BACKUP | "
                f"Round {round_id} | "
                f"Cluster {cluster_id} | "
                f"Client {client_id} | "
                f"Loss = {loss:.4f}"
            )


    print("Trust scores before evaluation:")
    print(server.trust_scores)

    server.trust_evaluation_and_backup_replacement()

    print("Trust scores after evaluation:")
    print(server.trust_scores)

    print("Accepted clients:")
    print(server.accepted_clients)


    global_model = server.aggregate(
        global_model,
        trim_ratio=0.2
    )
    val_loss, val_accuracy = evaluate_validation(
        global_model,
        val_dataset
    )

    print(
        f"Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Model relative change: "
        f"{server.model_relative_change:.8f}"
    )



    if server.model_relative_change < MODEL_CHANGE_THRESHOLD:

        val_loss, val_accuracy = evaluate_validation(
            global_model,
            val_dataset
        )

        print(
            f"Validation Loss: {val_loss:.6f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if previous_val_loss is not None:

            val_loss_change = abs(
                val_loss - previous_val_loss
            )

            print(
                f"Validation Loss Change: "
                f"{val_loss_change:.6f}"
            )

            if val_loss_change < VAL_LOSS_CHANGE_THRESHOLD:
                stable_checks += 1
            else:
                stable_checks = 0
            print(
                f"Stable checks: "
                f"{stable_checks}/{PATIENCE}"
            )


        previous_val_loss = val_loss

        if stable_checks >= PATIENCE:
            print(
                "Training stopped: "
                "convergence criteria satisfied."
            )
            break
    else:
        stable_checks = 0
        previous_val_loss = None

    print(f"Round {round_id} completed.")
    print("---------")
    

