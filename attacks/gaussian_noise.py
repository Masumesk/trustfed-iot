import numpy as np

def gaussian_noise_attack(update, strength=0.05):
    mean = 0.1
    variance = 0.1

    noise = np.random.normal(
        loc=mean,
        scale=np.sqrt(variance),
        size=update.shape,
    ).astype(
        update.dtype,
        copy=False,
    )

    return update + strength * noise