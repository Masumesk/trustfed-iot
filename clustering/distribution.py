#client
import numpy as np

def compute_histogram(dataset, indices, num_classes=10):

    targets = np.array(dataset.targets)
    labels = targets[indices]

    histogram= np.bincount(
        labels,
        minlength=num_classes
    )

    return histogram


def normalize_histogram(histogram,num_samples):

    if num_samples == 0:
        return np.zeros_like(histogram, dtype=float)

    return (histogram/num_samples)


def get_client_distribution(dataset, indices, num_samples, num_classes=10,):

    histogram = compute_histogram(dataset,indices,num_classes)
    distribution = normalize_histogram(histogram,num_samples)

    return distribution