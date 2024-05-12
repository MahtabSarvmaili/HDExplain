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

    train_loader, test_loader, class_names = real_data[args.data](n_test=10, subsample=True)

    explainer.data_influence(train_loader, cache=True)

    name_template = "{0}-{1}-KSD-SIM.pdf"

    idxs = np.random.choice(len(train_loader.dataset), 5, replace=False)
    for i in idxs:
        X_test, y_test = train_loader.dataset.__getitem__(i)
        X_test = X_test.unsqueeze(dim=0).cpu().numpy()
        y_test = np.array(y_test).reshape(1)
        influence_scores = explainer.ksd_debug(train_loader, X_test, y_test)
        dt_ids = influence_scores.argsort()[::-1].reshape(-1)
        ksd_eval_res = explainer.datapoint_model_discrepancy(train_loader, X_test, y_test, i, dt_ids)
        process_kernel_debugging(train_loader, i, dt_ids, ksd_eval_res, 0.99, class_names, name_template.format(args.data, i))


    # y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().numpy()
    #
    #
    #
    # X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
    # if args.gpu:
    #     X_test_tensor = X_test_tensor.cuda()
    # y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().cpu().numpy()
    #
    # top_explaination, influence_scores = explainer.pred_explanation(train_loader, X_test, topK=3)
    #
    # instances = explain_instance(train_loader.dataset, X_test_tensor, y_hat, top_explaination)
    #
    # if args.scale:
    #     name_template = "{0}*-{1}-{2}.pdf"
    # else:
    #     name_template = "{0}-{1}-{2}.pdf"
    # for i in range(len(instances)):
    #     plot_explanation_images(instances[i], class_names,
    #                             name=name_template.format(args.explainer, args.data, i))
    # # ------------------------------------------------------------------------------------------------ #
    # if args.synthetic:
    #     model = networks[args.network](feature_dim=2,
    #                                    latent_dim=10,
    #                                    n_classes=args.n_classes)
    #
    #     model.load_state_dict(
    #         torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network,
    #                                                        args.data,
    #                                                        args.n_classes)))
    #     explainer = explainers["YADEA"](model, args.n_classes)
    #
    #     X, y = synthetic_data[args.data](n_samples=500, n_classes=args.n_classes)
    #
    #     dataset = CustomDataset(np.array(X, dtype=np.float32), np.array(y, dtype=np.int_))
    #     dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    #
    #     explainer.data_influence(dataloader, cache=True)
    #
    #     idxs = np.random.choice(len(X), 5, replace=False)
    #     for i in idxs:
    #         X_test, y_test = X[i], y[i]
    #
    #         X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
    #         _, influence_scores = explainer.ksd_debug(dataloader, X_test, y_test, topK=3)
    #
    #     y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().numpy()
    #
    #
    #     # should use the explainer.pred_explanation for each datapoint
    #     # plot the returned ksd
    #     # based on the ksd, sort the datapoints and select the most/least relevant/irrelevant datapoints
    #
    #     bound = np.max([np.abs(np.min(influence_scores)), np.abs(np.max(influence_scores))])
    #
    #     if args.visualize:
    #         for index, test_point in enumerate(X_test):
    #             ksd_influence(X, y, test_point, y_hat[index],
    #                           influence_scores[index],
    #                           name="{0}-{1}-{2}.pdf".format(args.explainer, args.data, index),
    #                           clip=(-bound, bound)
    #                           )


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