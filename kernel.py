import torch
import pandas as pd
import numpy as np
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, save_dataframe_csv, load_dataframe_csv
from viz import ksd_influence
import argparse

from experiments import perturbation_explanation
import time
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
        dataloader, _, _ = real_data[args.data](n_test=10, subsample=True)
        
    st = time.time()

    explainer = explainers[args.explainer](model, args.n_classes, gpu=args.gpu, scale=args.scale, kernel=args.kernel)

    explainer.data_influence(dataloader, cache=True)

    coverage, hit_rate = perturbation_explanation(explainer, dataloader, seed=args.seed)
    et = time.time()

    try:
        df = load_dataframe_csv("tables", "Explainability_kernel_{0}.csv".format(args.data))
    except:
        columns = ['data', 'kernel', 'scale', 'coverage', 'hit_rate', 'execution_time']
        df = pd.DataFrame(columns=columns)

    results = dict()
    results['data'] = args.data
    results['kernel'] = args.kernel
    results['scale'] = args.scale
    results['coverage'] = coverage
    results['hit_rate'] = hit_rate
    results['execution_time'] = et - st
    df = df._append(results, ignore_index=True)
    save_dataframe_csv(df, "tables", "Explainability_kernel_{0}.csv".format(args.data))
        

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="SVHN")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-seed', dest='seed', type=check_int_positive, default=42)
    parser.add_argument('-explainer', dest='explainer', default="HDEXPLAIN")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.add_argument('-kernel', dest='kernel', default="RBF")
    args = parser.parse_args()
    main(args)