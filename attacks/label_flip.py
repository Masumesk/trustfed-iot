import torch


def label_flip_attack(labels, num_classes=10):
    return (labels + 1) % num_classes