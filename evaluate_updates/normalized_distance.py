import numpy as np


def compute_A_i(update, reference, reference_updates):

    distance = np.sqrt(np.sum((update - reference) ** 2))

    distances = []

    for ref_update in reference_updates:

        d = np.sqrt(np.sum((ref_update - reference)**2))
        distances.append(d)


    median_distance = np.median(distances)
    
    A_i = ( distance / (median_distance + (1e-8)) )

    return A_i