import torch
import numpy as np
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, explain_instance
from viz import ksd_influence
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset
from viz import plot_explanation_images


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

    X_test, y_test = next(iter(test_loader))

    explainer.data_influence(train_loader, cache=True)

    X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
    if args.gpu:
        X_test_tensor = X_test_tensor.cuda()
    y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().cpu().numpy()

    X_test = X_test[6:8]
    X_test_tensor = X_test_tensor[6:8]
    y_hat = y_hat[6:8]

    boxes = [[45,85,35,75],[10,50,40,80]]

    top_explaination, influence_scores = explainer.pred_explanation_with_cropping(train_loader, X_test, boxes, topK=3)

    instances = explain_instance(train_loader.dataset, X_test_tensor, y_hat, top_explaination)

    if args.scale:
        name_template = "{0}*-{1}-{2}_cropped.pdf"
    else:
        name_template = "{0}-{1}-{2}_cropped.pdf"
    for i in range(len(instances)):
        plot_explanation_images(instances[i], class_names,
                                name=name_template.format(args.explainer, args.data, i))
        

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="MRI")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=4)
    parser.add_argument('-explainer', dest='explainer', default="HDEXPLAIN")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    args = parser.parse_args()
    main(args)