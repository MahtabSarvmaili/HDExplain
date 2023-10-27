import torch
import pandas as pd
import numpy as np
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, save_dataframe_csv, load_dataframe_csv
from viz import ksd_influence
import argparse

from experiments import perturbation_explanation

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset


def main(args):
    model = networks[args.network](feature_dim=2, 
                                    latent_dim=10, 
                                    n_classes=args.n_classes)
    model.load_state_dict(
        torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network, 
                                                    args.data, 
                                                    args.n_classes)))
    explainer = explainers[args.explainer](model, args.n_classes, gpu=args.gpu, scale=args.scale)

    if args.synthetic:
        X,y = synthetic_data[args.data](n_samples=500, n_classes = args.n_classes)
        dataset = CustomDataset(np.array(X, dtype=np.float32), np.array(y, dtype=np.int_))
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    else:
        dataloader, _, _ = real_data[args.data](n_test_sample=10)

    explainer.data_influence(dataloader, cache=True)

    try:
        df = load_dataframe_csv("tables", "Explainability_{0}.csv".format(args.data))
    except:
        columns = ['data', 'explainer', 'coverage', 'hit_rate']
        df = pd.DataFrame(columns=columns)

    seed = np.random.randint(100)
    coverage, hit_rate = perturbation_explanation(explainer, dataloader, seed=seed)
    results = dict()
    results['data'] = args.data
    results['explainer'] = args.explainer
    results['seed'] = seed
    results['coverage'] = coverage
    results['hit_rate'] = hit_rate
    df = df.append(results, ignore_index=True)
    save_dataframe_csv(df, "tables", "Explainability_{0}.csv".format(args.data))
        

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