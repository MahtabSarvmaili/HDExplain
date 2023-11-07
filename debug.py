import torch
import numpy as np
from utils import explainers, networks, synthetic_data, check_int_positive
from viz import data_debug_2d
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset


def main(args):
    if args.synthetic:
        model = networks[args.network](feature_dim=2, 
                                       latent_dim=10, 
                                       n_classes=args.n_classes)
        
        trainer = ClassifierTrainer(model)

        X,y = synthetic_data[args.data](n_samples=500, n_classes = args.n_classes)

        np.random.seed(42)

        corrupt_index = np.random.choice(len(y), np.rint(len(y)/10).astype(int))

        np.random.seed()

        y_corrupted = np.array(y, copy=True)  
        y_corrupted[corrupt_index] = 1 - y[corrupt_index]

        dataset = CustomDataset(np.array(X, dtype=np.float32), np.array(y_corrupted, dtype=np.int_))
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        trainer.train(model, dataloader, 
                    save=False,
                    epochs=200)

        explainer = explainers[args.explainer](model, args.n_classes)

        explainer.data_influence(dataloader, cache=True)

        scores, order = explainer.data_debugging(dataloader)

        groundtruth = np.zeros_like(scores)
        groundtruth[corrupt_index] = 1

        hard_decision = np.zeros_like(y)

        hard_decision[order[-np.rint(len(y)/10).astype(int):]] = 1

        if args.visualize:
            data_debug_2d(X, y, (hard_decision, groundtruth), 
                          name="debug-{0}-{1}.pdf".format(args.explainer, args.data),
                              )


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Debug")
    parser.add_argument('-network', dest='network', default="SimpleNet")
    parser.add_argument('-data', dest='data', default="Moon")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=2)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    args = parser.parse_args()
    main(args)