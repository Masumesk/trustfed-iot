import numpy as np


def gaussian_noise_attack(update, strength=1.0):


    update = np.asarray(update)

    update_std = np.std(update)

    if update_std < 1e-12:
        update_std = 1e-12

    noise = np.random.normal(
        loc=0.0,
        scale=strength * update_std,
        size=update.shape,
    ).astype(
        update.dtype,
        copy=False,
    )

    return update + noise