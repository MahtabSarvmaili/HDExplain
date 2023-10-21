from explainers import *
from models.classifiers import SimpleNet, resnet18
from dataloaders import rectangular, moon


explainers = {
    "YADEA": KSDExplainer,
    "IF": InfluenceFunction
}

networks = {
    "SimpleNet": SimpleNet,
    "ResNet": resnet18
}

synthetic_data = {
    "Moon": moon,
    "Rectangular": rectangular
}