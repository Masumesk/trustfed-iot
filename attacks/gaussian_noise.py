import numpy as np


def gaussian_noise_attack(update, sigma=1.0):

    noise = np.random.normal(
        0,
        sigma,
        size=update.shape
    )

    return update + noise