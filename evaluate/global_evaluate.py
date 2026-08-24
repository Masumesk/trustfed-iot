import torch
from torch.utils.data import DataLoader


def evaluate_model(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0


    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    return {
        "accuracy": correct / total,
        "loss": total_loss / len(loader)
    }