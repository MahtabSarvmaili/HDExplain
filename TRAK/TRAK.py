import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import explainers, networks, synthetic_data, real_data, check_int_positive, explain_instance, save_dataframe_csv, load_dataframe_csv
from viz import ksd_influence
import argparse
from trak import TRAKer
from torch.utils.data import DataLoader
from models import ClassifierTrainer, CustomDataset
from viz import plot_explanation_images
from experiments.explainability import perturbation


def hit(vector_true_dense, vector_predict):
    hits = np.isin(vector_predict, vector_true_dense)
    return hits

def recallk(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_true_dense)


def precisionk(vector_predict, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_predict)

def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcgk(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg



def main(args):
    model = networks[args.network](num_classes=args.n_classes)
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.training:
        for x in range(3):
            train_loader, test_loader, class_names = real_data[args.data](n_test=10, random_state=x)
            model = networks[args.network](num_classes=args.n_classes)
            trainer = ClassifierTrainer(model)
            trainer.train(model, train_loader, 
                        name="{0}-{1}-{2}-{3}-TRAK".format(args.network, 
                                                    args.data, 
                                                    args.n_classes, x), 
                        epochs=100)
    else:
        model.load_state_dict(
            torch.load("checkpoints/{0}-{1}-{2}.pt".format(args.network, 
                                                        args.data, 
                                                        args.n_classes)))
    model.to(device)
    train_loader, test_loader, class_names = real_data[args.data](n_test=10)
    checkpoints = ["checkpoints/{0}-{1}-{2}-{3}-TRAK.pt".format(args.network, args.data, args.n_classes, x) for x in range(3)]
    checkpoints = [torch.load(ckpt, map_location=device) for ckpt in checkpoints]

    traker = TRAKer(model=model, task='image_classification', train_set_size=train_loader.dataset.__len__())
    if args.trak_score:
        for model_id, ckpt in enumerate(tqdm(checkpoints)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
            traker.load_checkpoint(ckpt, model_id=model_id)

            for batch in train_loader:
                batch = [x.cuda() for x in batch]
                # TRAKer computes features corresponding to the batch of examples,
                # using the checkpoint loaded above.
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        # Tells TRAKer that we've given it all the information, at which point
        # TRAKer does some post-processing to get ready for the next step
        # (scoring target examples).
        traker.finalize_features()
    if args.trak_score_test:
        for model_id, ckpt in enumerate(tqdm(checkpoints)):
            traker.start_scoring_checkpoint(exp_name='quickstart',
                                            checkpoint=ckpt,
                                            model_id=model_id,
                                            num_targets=len(test_loader.dataset))
            for batch in test_loader:
                batch = [x.cuda() for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='quickstart')
    
    # perturbation
    perturbed_dataloader, corrupt_index = perturbation(original_dataloader=train_loader, size=100, seed=1, flip=False, gpu=False)

    if args.trak_score_test_perturbation:
        for model_id, ckpt in enumerate(tqdm(checkpoints)):
            traker.start_scoring_checkpoint(exp_name='quickstart_perturbation',
                                            checkpoint=ckpt,
                                            model_id=model_id,
                                            num_targets=len(perturbed_dataloader.dataset))
            for batch in perturbed_dataloader:
                batch = [x.cuda() for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='quickstart_perturbation')
    from numpy.lib.format import open_memmap
    from experiments.debug import data_debugging_NO_EXP

    _scores_p = open_memmap('./TRAK/trak_results/scores/quickstart_perturbation.mmap')
    _scores_p = _scores_p.T
    order_p = np.argsort(_scores_p, axis=1)[:, ::-1]
    scores_p = np.sort(_scores_p, axis=1)[:, ::-1]
    
    _scores = open_memmap('./TRAK/trak_results/scores/quickstart.mmap')
    _scores = _scores.T
    order = np.argsort(_scores, axis=1)[:, ::-1]
    scores = np.sort(_scores, axis=1)[:, ::-1]
    order_ = order[corrupt_index, :100]

    X_test_tensor = train_loader.dataset.__getitem__(corrupt_index[0])[0]
    X_test_tensor = torch.from_numpy(np.array(X_test_tensor, dtype=np.float32))
    X_test_tensor = X_test_tensor.cuda()
    X_test_tensor = X_test_tensor.unsqueeze(0)
    y_hat = torch.argmax(model.predict(X_test_tensor), dim=1).detach().cpu().numpy()

    instances = explain_instance(train_loader.dataset, X_test_tensor, y_hat, [order_p[0, :3]])
    name_template = "{0}-{1}-{2}-TRAK.pdf"
    for i in range(len(instances)):
        plot_explanation_images(instances[i], class_names, 
                                name=name_template.format(args.explainer, args.data, i))
        
    recall = []
    precision = []
    ndcg = []
    ks = list(range(10, 110, 10))

    for k in ks:
        recall_k = []
        precision_k = []
        ndcg_k = []
        for i in range(100):
            hits = hit(order[i], order_p[i, :k])
            recall_k.append(recallk(order[i], hits))
            precision_k.append(precisionk(order_p[i, :k], hits))
            ndcg_k.append(ndcgk(corrupt_index, order_p[i, :k], hits))
        recall.append(np.mean(recall_k))
        precision.append(np.mean(precision_k))
        ndcg.append(np.mean(ndcg_k))
    
    columns = ['data', 'explainer', 'scale', 'k', 'recall', 'precision', 'ndcg']
    df = pd.DataFrame(columns=columns)

    results_list = []
    for i, k in enumerate(ks):
        results = dict()
        results['data'] = args.data
        results['explainer'] = 'TRAK'
        results['scale'] = args.scale
        results['k'] = k
        results['recall'] = recall[i]
        results['precision'] = precision[i]
        results['ndcg'] = ndcg[i]
        results_list.append(results)
    df = pd.concat([df, pd.DataFrame(results_list)], ignore_index=True)
    save_dataframe_csv(df, "tables", "Debug_{0}_TRAK.csv".format(args.data))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Explain")
    parser.add_argument('-network', dest='network', default="ResNet")
    parser.add_argument('-data', dest='data', default="CIFAR10")
    parser.add_argument('-n_classes', dest='n_classes', type=check_int_positive, default=10)
    parser.add_argument('-explainer', dest='explainer', default="YADEA")
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    parser.add_argument('--scale', dest='scale', action='store_true', default=True)
    parser.add_argument('--training', dest='training', action='store_true', default=False)
    parser.add_argument('--trak_score', dest='trak_score', action='store_true', default=False)
    parser.add_argument('--trak_score_test', dest='trak_score_test', action='store_true', default=False)
    parser.add_argument('--trak_score_test_perturbation', dest='trak_score_test_perturbation', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
