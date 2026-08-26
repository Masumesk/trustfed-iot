import numpy as np
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
    total_loss = 0.0

    class_correct = None
    class_total = None

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

           
            num_classes = outputs.shape[1]

            if class_correct is None:

                class_correct = np.zeros(
                    num_classes,
                    dtype=np.int64
                )

                class_total = np.zeros(
                    num_classes,
                    dtype=np.int64
                )

            
            for class_id in range(num_classes):

                mask = labels == class_id

                class_total[class_id] += int(
                    mask.sum().item()
                )

                class_correct[class_id] += int(
                    (
                        (predictions == labels)
                        & mask
                    ).sum().item()
                )

   

    accuracy = correct / total

   

    per_class_accuracy = {}

    for class_id in range(
        len(class_total)
    ):

        if class_total[class_id] > 0:

            per_class_accuracy[
                str(class_id)
            ] = (
                class_correct[class_id]
                /
                class_total[class_id]
            )

    class_accuracies = np.array(
        list(
            per_class_accuracy.values()
        ),
        dtype=float
    )

   
    balanced_accuracy = float(
        np.mean(class_accuracies)
    )

  

    worst_class_accuracy = float(
        np.min(class_accuracies)
    )

    

    class_accuracy_std = float(
        np.std(class_accuracies)
    )

    

    return {

        "accuracy":
            float(accuracy),

        "loss":
            float(
                total_loss
                /
                len(loader)
            ),

        "balanced_accuracy":
            balanced_accuracy,

        "worst_class_accuracy":
            worst_class_accuracy,

        "class_accuracy_std":
            class_accuracy_std,

        "per_class_accuracy":
            per_class_accuracy,
    }