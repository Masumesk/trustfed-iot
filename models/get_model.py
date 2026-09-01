from config import DATASET
from models.mnist_model import MNISTCNN
from models.cifar10_model import CIFAR10CNN


def get_model():

    if DATASET=="MNIST":
        return MNISTCNN()

    elif DATASET=="CIFAR10":
        return CIFAR10CNN()