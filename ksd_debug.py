import pandas as pd
import torch
import numpy as np
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, explain_instance
from viz import ksd_influence
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset
from viz import plot_explanation_images
from utils.ksd_debugging import process_kernel_debugging


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
    infl_fun = explainers["IF"](model, args.n_classes, gpu=args.gpu, scale=args.scale)

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

    class_losses = df.groupby('Class')['Loss'].mean()
    min_loss_indices = df.groupby('Class')['Loss'].idxmin()
    max_loss_indices = df.groupby('Class')['Loss'].idxmax()

    explainer.data_influence(train_loader, cache=True)

    name_template = "{0}-{1}-KSD-SIM_rand.pdf"

    for i in np.random.choice(len(train_loader.dataset), 20, replace=False):
        X_test, y_test = train_loader.dataset.__getitem__(i)
        X_test = X_test.unsqueeze(dim=0).cpu().numpy()
        y_test = np.array(y_test).reshape(1)
        influence_scores = explainer.ksd_debug(train_loader, X_test, y_test)
        dt_ids = influence_scores.argsort()[::-1].reshape(-1)
        ksd_eval_res = explainer.datapoint_model_discrepancy(train_loader, X_test, y_test, i, dt_ids)
        process_kernel_debugging(train_loader, model, i, dt_ids, ksd_eval_res, 0.7, 0.7, class_names,
                                 name_template.format(args.data, i))

    name_template = "{0}-{1}-KSD-SIM_min.pdf"

    for i in min_loss_indices:
        X_test, y_test = train_loader.dataset.__getitem__(i)
        X_test = X_test.unsqueeze(dim=0).cpu().numpy()
        y_test = np.array(y_test).reshape(1)
        influence_scores = explainer.ksd_debug(train_loader, X_test, y_test)
        dt_ids = influence_scores.argsort()[::-1].reshape(-1)
        ksd_eval_res = explainer.datapoint_model_discrepancy(train_loader, X_test, y_test, i, dt_ids)
        process_kernel_debugging(train_loader, model, i, dt_ids, ksd_eval_res, 0.99, 0.7, class_names,
                                 name_template.format(args.data, i))

    name_template = "{0}-{1}-KSD-SIM_max.pdf"

    for i in max_loss_indices:
        X_test, y_test = train_loader.dataset.__getitem__(i)
        X_test = X_test.unsqueeze(dim=0).cpu().numpy()
        y_test = np.array(y_test).reshape(1)
        influence_scores = explainer.ksd_debug(train_loader, X_test, y_test)
        dt_ids = influence_scores.argsort()[::-1].reshape(-1)
        ksd_eval_res = explainer.datapoint_model_discrepancy(train_loader, X_test, y_test, i, dt_ids)
        process_kernel_debugging(train_loader, model, i, dt_ids, ksd_eval_res, 0.99, 0.7, class_names,
                                 name_template.format(args.data, i))



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="CIFAR10")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    args = parser.parse_args()
    main(args)