import torch
import numpy as np
import pandas as pd
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, save_dataframe_csv, load_dataframe_csv
from viz import data_debug_2d
import argparse
from experiments import data_debugging

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset


def main(args):

    if args.synthetic:
        model = networks[args.network](feature_dim=2, 
                                        latent_dim=10, 
                                        n_classes=args.n_classes)
    else:
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

    if args.synthetic:
        X,y = synthetic_data[args.data](n_samples=500, n_classes = args.n_classes)
        dataset = CustomDataset(np.array(X, dtype=np.float32), np.array(y, dtype=np.int_))
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    else:
        dataloader, _, _ = real_data[args.data](n_test=10, subsample=False)

    if args.synthetic:
        
        trainer = ClassifierTrainer(model)

        trainer.train(model, dataloader, 
                    save=False,
                    epochs=200)

    explainer = explainers[args.explainer](model, args.n_classes, gpu=args.gpu, scale=args.scale)

    ks = list(range(10, 110, 10))

    recall, precision, ndcg = data_debugging(explainer, dataloader, 
                                                args.n_classes, n_corrputed=100, 
                                                ks=ks, 
                                                seed=args.seed, gpu=args.gpu, subsample=args.subsample)
    
    try:
        df = load_dataframe_csv("tables", "Debug_{0}.csv".format(args.data))
    except:
        columns = ['data', 'explainer', 'scale', 'k', 'recall', 'precision', 'ndcg']
        df = pd.DataFrame(columns=columns)

    results_list = []
    for i, k in enumerate(ks):
        results = dict()
        results['data'] = args.data
        results['explainer'] = args.explainer
        results['scale'] = args.scale
        results['k'] = k
        results['recall'] = recall[i]
        results['precision'] = precision[i]
        results['ndcg'] = ndcg[i]
        results_list.append(results)
    df = pd.concat([df, pd.DataFrame(results_list)], ignore_index=True)
    save_dataframe_csv(df, "tables", "Debug_{0}.csv".format(args.data))

        # explainer.data_influence(dataloader, cache=True)

        # scores, order = explainer.data_debugging(dataloader)

        # groundtruth = np.zeros_like(scores)
        # groundtruth[corrupt_index] = 1

        # hard_decision = np.zeros_like(y)

        # hard_decision[order[-np.rint(len(y)/10).astype(int):]] = 1

        # if args.visualize:
        #     data_debug_2d(X, y, (hard_decision, groundtruth), 
        #                   name="debug-{0}-{1}.pdf".format(args.explainer, args.data),
        #                       )


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Debug")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="CIFAR10")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-seed', dest='seed', type=check_int_positive, default=42)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--subsample', dest='subsample', action='store_true')
    args = parser.parse_args()
    main(args)