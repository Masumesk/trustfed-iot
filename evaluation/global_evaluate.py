import torch
from torch.utils.data import DataLoader


def evaluate_model(model, dataset, batch_size=256):

    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    # Detect model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Sum loss over samples so average_loss is independent of batch size
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    was_training = model.training
    model.eval()

    total_loss = torch.zeros((),device=device,dtype=torch.float64,)
    confusion_matrix = None

    num_classes = None

    loss_is_finite = True
    outputs_are_finite = True

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            if not torch.isfinite(outputs).all().item():
                outputs_are_finite = False
                break

            if num_classes is None:
                num_classes = outputs.shape[1]

            # Loss
            batch_loss = criterion(outputs, labels)
            if torch.isfinite(batch_loss).item():
                total_loss += batch_loss.detach().to(torch.float64)

                if not torch.isfinite(total_loss).item():
                    loss_is_finite = False
            else:
                loss_is_finite = False

            # Predictions
            predictions = outputs.argmax(dim=1)

            # Initialize confusion matrix once
            if confusion_matrix is None:
                confusion_matrix = torch.zeros(
                    (num_classes, num_classes),
                    dtype=torch.long,
                    device=device,
                )

            # Fast confusion-matrix update
            # row = true class, column = predicted class
            indices = labels * num_classes + predictions

            batch_confusion = torch.bincount(
                indices,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            confusion_matrix += batch_confusion

    # Restore original model state
    if was_training:
        model.train()

    if not outputs_are_finite:
        return {
            "accuracy": None,
            "loss": None,
            "balanced_accuracy": None,
            "worst_class_accuracy": None,
            "class_accuracy_std": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "per_class_accuracy": {},
            "status": "non_finite_outputs",
        }

    confusion_matrix = confusion_matrix.cpu()

    # Per-class statistics
    class_total = confusion_matrix.sum(dim=1).float()
    class_correct = confusion_matrix.diagonal().float()

    total_samples = int(class_total.sum().item())
    total_correct = float(class_correct.sum().item())

    # Global accuracy
    accuracy = total_correct / total_samples

    # Average sample loss
    if loss_is_finite:
        average_loss = total_loss.item() / total_samples
    else:
        average_loss = None

    # Classes present in evaluation data
    valid_classes = class_total > 0

    class_accuracies = class_correct[valid_classes] / class_total[valid_classes]

    # Per-class accuracy
    per_class_accuracy = {}

    for class_id in range(len(class_total)):
        if class_total[class_id] > 0:
            per_class_accuracy[str(class_id)] = float(
                (class_correct[class_id] / class_total[class_id]).item()
            )

    # Balanced accuracy
    balanced_accuracy = float(class_accuracies.mean().item())

    # Worst-class accuracy
    worst_class_accuracy = float(class_accuracies.min().item())

    # Standard deviation of class accuracies
    class_accuracy_std = float(class_accuracies.std(unbiased=False).item())

    # Precision / Recall / F1
    true_positive = confusion_matrix.diagonal().float()
    predicted_total = confusion_matrix.sum(dim=0).float()
    actual_total = confusion_matrix.sum(dim=1).float()

    precision_per_class = true_positive / predicted_total.clamp(min=1)
    recall_per_class = true_positive / actual_total.clamp(min=1)
    f1_per_class = (2 * precision_per_class* recall_per_class/ (precision_per_class + recall_per_class).clamp(min=1e-12))
    macro_precision = float(precision_per_class[valid_classes].mean().item())
    macro_recall = float(recall_per_class[valid_classes].mean().item())
    macro_f1 = float(f1_per_class[valid_classes].mean().item())

    # Weighted F1
    class_weights = actual_total / actual_total.sum()

    weighted_f1 = float((f1_per_class * class_weights).sum().item())

    return {
        "accuracy": float(accuracy),
        "loss": None if average_loss is None else float(average_loss),
        "balanced_accuracy": balanced_accuracy,
        "worst_class_accuracy": worst_class_accuracy,
        "class_accuracy_std": class_accuracy_std,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_accuracy,
        "status": "ok" if loss_is_finite else "non_finite_loss",
    }
