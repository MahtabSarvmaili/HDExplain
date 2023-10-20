from explainers import *
from models.classifiers import SimpleNet
from dataloaders import rectangular, moon

explainers = {
    "YADEA": KSDExplainer
}

networks = {
    "SimpleNet": SimpleNet
}

synthetic_data = {
    "Moon": moon,
    "Rectangular": rectangular
}