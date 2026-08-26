import torch
from torch.utils.data import DataLoader
from torch import nn


def evaluate_validation(
    model,
    dataset,
    batch_size=128
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            batch_samples = labels.size(0)

            total_loss += (
                loss.item()
                * batch_samples
            )

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_samples

    validation_loss = (
        total_loss / total_samples
    )

    validation_accuracy = (
        total_correct / total_samples
    )

    if was_training:
        model.train()

    return (
        validation_loss,
        validation_accuracy
    )