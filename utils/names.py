from explainers import *
from models.classifiers import SimpleNet
from dataloaders import rectangular, moon

explainer_names = {
    "YADEA": KSDExplainer
}

classifier_names = {
    "SimpleNet": SimpleNet
}

synthetic_data_names = {
    "Moon": moon,
    "Rectangular": rectangular
}