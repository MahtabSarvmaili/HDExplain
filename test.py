import torch
import numpy as np
from utils import explainers, networks, synthetic_data, check_int_positive
from viz import ksd_influence
import argparse

from experiments import perturbation_explanation

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset


def main(args):
    if args.synthetic:
        model = networks[args.network](feature_dim=2, 
                                       latent_dim=10, 
                                       n_classes=args.n_classes)
        
        model.load_state_dict(
            torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network, 
                                                        args.data, 
                                                        args.n_classes)))
        explainer = explainers[args.explainer](model, args.n_classes)

        X,y = synthetic_data[args.data](n_samples=500, n_classes = args.n_classes)

        dataset = CustomDataset(np.array(X, dtype=np.float32), np.array(y, dtype=np.int_))
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        explainer.data_influence(dataloader, cache=True)

        val = perturbation_explanation(model, explainer, dataloader)

        import ipdb; ipdb.set_trace()
        

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="SimpleNet")
    parser.add_argument('-data', dest='data', default="Moon")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=2)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    args = parser.parse_args()
    main(args)