from explainers import *
from models.classifiers import SimpleNet, resnet18
from models.vit import vit_model
from dataloaders import rectangular, moon, cifar10, ocea, mri, svhn, cifar10_224, svhn_224


explainers = {
    "HDEXPLAIN": KSDExplainer,
    "IF": InfluenceFunction,
    "TracIn": TracIn,
    "RPS": RepresenterPointSelection
}

networks = {
    "SimpleNet": SimpleNet,
    "ResNet": resnet18,
    "ViT": vit_model
}

synthetic_data = {
    "Moon": moon,
    "Rectangular": rectangular
}

real_data = {
    "CIFAR10": cifar10,
    "OCEA": ocea,
    "MRI": mri,
    "SVHN": svhn,
    "CIFAR10_224": cifar10_224,
    "SVHN_224": svhn_224
}