from explainers import *
from models.classifiers import SimpleNet, resnet18
from dataloaders import rectangular, moon, cifar10, ocea


explainers = {
    "YADEA": KSDExplainer,
    "IF": InfluenceFunction,
    "TracIn": TracIn,
    "RPS": RepresenterPointSelection
}

networks = {
    "SimpleNet": SimpleNet,
    "ResNet": resnet18
}

synthetic_data = {
    "Moon": moon,
    "Rectangular": rectangular
}

real_data = {
    "CIFAR10": cifar10,
    "OCEA": ocea
}