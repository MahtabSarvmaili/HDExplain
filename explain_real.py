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
    test_iter = iter(test_loader)
    X_test, y_test = next(test_iter)

    explainer.data_influence(train_loader, cache=True)

    X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
    if args.gpu:
        X_test_tensor = X_test_tensor.cuda()
    y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().cpu().numpy()
    
    top_explaination, influence_scores = explainer.pred_explanation(train_loader, X_test, topK=3)

    instances = explain_instance(train_loader.dataset, X_test_tensor, y_hat, top_explaination)

    if args.scale:
        name_template = "{0}*-{1}-{2}.pdf"
    else:
        name_template = "{0}-{1}-{2}.pdf"
    for i in range(len(instances)):
        plot_explanation_images(instances[i], class_names, 
                                name=name_template.format(args.explainer, args.data, i))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="SVHN")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true', default=True)
    args = parser.parse_args()
    main(args)