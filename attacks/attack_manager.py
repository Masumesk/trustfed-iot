# from attacks.label_flip import label_flip_attack
from attacks.gaussian_noise import gaussian_noise_attack
from attacks.sign_flip import sign_flip_attack
from attacks.scaling import scaling_attack


def apply_attack(update, attack_type):

    if attack_type in (None, "label_flip"):
        return update


    # if attack_type == "label_flip":
    #     return label_flip_attack(update)


    elif attack_type == "gaussian":
        return gaussian_noise_attack(update)


    elif attack_type == "sign_flip":
        return sign_flip_attack(update)


    elif attack_type == "scaling":
        return scaling_attack(update)


    else:
        raise ValueError(
            f"Unknown attack: {attack_type}"
        )