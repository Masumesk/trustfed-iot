from attacks.label_flip import label_flip_attack
from attacks.gaussian_noise import gaussian_noise_attack
from attacks.sign_flip import sign_flip_attack
from attacks.scaling import scaling_attack


def apply_attack(model, attack_type):

    if attack_type is None:
        return model


    if attack_type == "label_flip":
        return label_flip_attack(model)


    elif attack_type == "gaussian":
        return gaussian_noise_attack(model)


    elif attack_type == "sign_flip":
        return sign_flip_attack(model)


    elif attack_type == "scaling":
        return scaling_attack(model)


    else:
        raise ValueError(
            f"Unknown attack: {attack_type}"
        )