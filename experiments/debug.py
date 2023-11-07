import numpy as np
import torch

def data_debugging(explainer, dataloader, n_classes, n_corrputed, ks, seed=1, gpu=False):

    data_size = len(dataloader.dataset)
    np.random.seed(seed)
    corrupt_index = np.random.choice(data_size, n_corrputed, replace=False)
    corrupted_labels = dataloader.dataset.targets[corrupt_index] + 1
    corrupted_labels[corrupted_labels>=n_classes] = 0
    dataloader.dataset.targets[corrupt_index] = corrupted_labels

    explainer.data_influence(dataloader, cache=True)
    
    scores, order = explainer.data_debugging(dataloader)

    recall = []
    precision = []
    ndcg = []
    for k in ks:
        hits = hit(corrupt_index, order[:k])
        import ipdb; ipdb.set_trace()
        recall.append(recallk(corrupt_index, hits))
        precision.append(precisionk(order[:k], hits))
        ndcg.append(ndcgk(corrupt_index, order[:k], hits))

    return recall, precision, ndcg


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