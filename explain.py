import torch
from utils import explainers, networks, synthetic_data
import argparse


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

        explainer.data_influence(X, y, cache=True)

        X_test,_ = synthetic_data[args.data](n_samples=5, 
                                             n_classes = args.n_classes, 
                                             random_state=0)
        
        top_influencers = explainer.pred_explanation(X, X_test, topK=3)

        import ipdb;ipdb.set_trace()
        


        

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-network', dest='network', default="SimpleNet")
    parser.add_argument('-data', dest='data', default="Moon")
    parser.add_argument('-n_classes', dest='n_classes', default=2)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    args = parser.parse_args()
    main(args)