import torch
import numpy as np
from utils import explainers, networks, synthetic_data, check_int_positive
from viz import ksd_influence
import argparse

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

        X_test,_ = synthetic_data[args.data](n_samples=5, 
                                             n_classes = args.n_classes, 
                                             random_state=0)
        
        X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().numpy()
        
        _, influence_scores = explainer.pred_explanation(dataloader, X_test, topK=3)

        bound = np.max([np.abs(np.min(influence_scores)), np.abs(np.max(influence_scores))])

        if args.visualize:
            for index, test_point in enumerate(X_test):
                ksd_influence(X, y, test_point, y_hat[index], 
                              influence_scores[index], 
                              name="{0}-{1}-{2}.pdf".format(args.explainer, args.data, index),
                              clip=(-bound, bound)
                              )
        

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="SimpleNet")
    parser.add_argument('-data', dest='data', default="Moon")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=2)
    parser.add_argument('-explainer', dest='explainer', default="HDEXPLAIN")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    args = parser.parse_args()
    main(args)