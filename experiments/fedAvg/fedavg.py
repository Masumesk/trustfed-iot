import numpy as np


def fedavg_updates(updates, sample_counts):

    total_samples = sum(sample_counts)

    aggregated_update = np.zeros_like(updates[0])
    weighted_update = np.empty_like(updates[0])

    for update, num_samples in zip(updates, sample_counts):

        weight = num_samples / total_samples
        np.multiply(update, weight, out=weighted_update)
        np.add(aggregated_update, weighted_update, out=aggregated_update)

    return aggregated_update
