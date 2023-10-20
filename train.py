import numpy as np
import argparse

from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset

from utils import networks, synthetic_data

def main(args):
    if args.synthetic:
        X,y = synthetic_data[args.data](n_samples=500, n_classes = args.n_classes)
        dataset = CustomDataset(np.array(X, dtype=np.float32), y)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    
        model = networks[args.network](feature_dim=2, latent_dim=10, 
                                        n_classes=args.n_classes)
        trainer = ClassifierTrainer(model)
        trainer.train(model, dataloader, 
                    name="{0}-{1}-{2}".format(args.network, 
                                              args.data, 
                                              args.n_classes), 
                    epochs=100)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-network', dest='network', default="SimpleNet")
    parser.add_argument('-data', dest='data', default="Moon")
    parser.add_argument('-n_classes', dest='n_classes', default=2)
    parser.add_argument('--synthetic', dest='synthetic', action='store_true')
    args = parser.parse_args()
    main(args)