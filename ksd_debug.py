import pandas as pd
import torch
import numpy as np
from numpy.ma.core import indices

from utils import explainers, networks, synthetic_data, real_data, check_int_positive, explain_instance
from viz import ksd_influence
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset
from viz import plot_explanation_images
from utils.ksd_debugging import index_to_debug


def main(args):
    model = networks[args.network](num_classes=args.n_classes)

    try:
        model.load_state_dict(
            torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network,
                                                           args.data,
                                                           args.n_classes)))
    except:
        model.load_state_dict(
            torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network,
                                                           args.data,
                                                           args.n_classes))['net'])
    explainer = explainers[args.explainer](model, args.n_classes, gpu=args.gpu, scale=args.scale)
    train_loader, test_loader, class_names = real_data[args.data](n_test=10, subsample=True)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    loss_values = []
    classes = []

    model.eval()
    data_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False)
    index = 0
    for data in data_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_values.extend(loss.cpu().detach().numpy())
        classes.extend(labels.cpu().numpy())
        index += data_loader.batch_size

    data_points = list(zip(classes, loss_values))
    df = pd.DataFrame(data_points, columns=['Class', 'Loss'])

    min_loss_indices = df.groupby('Class')['Loss'].idxmin()
    max_loss_indices = df.groupby('Class')['Loss'].idxmax()

    explainer.data_influence(train_loader, cache=True)

    name_template = "{0}-{1}-KSD-SIM_rand.pdf"
    indices = np.random.choice(len(train_loader.dataset), 40, replace=False)
    index_to_debug(
        model, train_loader, indices, explainer, class_names, name_template, 0.7, 0.7, args)

    name_template = "{0}-{1}-KSD-SIM_min.pdf"
    index_to_debug(
        model, train_loader, min_loss_indices, explainer,
        class_names, name_template, 0.99, 0.7, args)

    name_template = "{0}-{1}-KSD-SIM_max.pdf"
    index_to_debug(
        model, train_loader, max_loss_indices, explainer,
        class_names, name_template, 0.99, 0.7, args)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="CIFAR10")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-explainer', dest='explainer', default="HDEXPLAIN")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    args = parser.parse_args()
    main(args)