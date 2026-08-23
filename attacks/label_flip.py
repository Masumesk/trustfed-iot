import torch


def label_flip_attack(label, num_classes=10):

    flipped_label = (label + 1) % num_classes

    return flipped_label