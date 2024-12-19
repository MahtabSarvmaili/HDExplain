import numpy as np
import torch

def data_debugging(explainer, dataloader, n_classes, n_corrputed, ks, seed=1, gpu=False, subsample=False):

    dataset = dataloader.dataset

    np.random.seed(seed)
    tens = np.random.choice(len(dataset), round(len(dataset)/10), replace=False).tolist()

    data_size = len(tens)

    np.random.seed(seed)
    corrupt_index = np.random.choice(data_size, min(n_corrputed, data_size), replace=False)
    backtrack_corrupted_index = np.array(tens)[corrupt_index]
    targets = np.array(dataset.targets)
    corrupted_labels = targets[backtrack_corrupted_index] + 1
    corrupted_labels[corrupted_labels>=n_classes] = 0

    targets[backtrack_corrupted_index] = corrupted_labels
    dataset.targets = targets.tolist()

    if subsample:
        dataset_1 = torch.utils.data.Subset(dataset, tens)

        newloader = torch.utils.data.DataLoader(
            dataset_1, batch_size=128, shuffle=False)
    else:
        newloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False)
        corrupt_index = backtrack_corrupted_index

    explainer.data_influence(newloader, cache=True)
    
    scores, order = explainer.data_debugging(newloader)

    recall = []
    precision = []
    ndcg = []
    for k in ks:
        hits = hit(corrupt_index, order[:k])
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


def data_debugging_NO_EXP(dataloader, scores, order, n_classes, n_corrputed, ks, seed=1, gpu=False, subsample=False):

    dataset = dataloader.dataset

    np.random.seed(seed)
    tens = np.random.choice(len(dataset), round(len(dataset)/10), replace=False).tolist()

    data_size = len(tens)

    np.random.seed(seed)
    corrupt_index = np.random.choice(data_size, min(n_corrputed, data_size), replace=False)
    backtrack_corrupted_index = np.array(tens)[corrupt_index]
    targets = np.array(dataloader.dataset.tensors[1])
    corrupted_labels = targets[backtrack_corrupted_index] + 1
    corrupted_labels[corrupted_labels>=n_classes] = 0

    targets[backtrack_corrupted_index] = corrupted_labels
    dataset.targets = targets.tolist()

    if subsample:
        dataset_1 = torch.utils.data.Subset(dataset, tens)

        newloader = torch.utils.data.DataLoader(
            dataset_1, batch_size=128, shuffle=False)
    else:
        newloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False)
        corrupt_index = backtrack_corrupted_index
    

    recall = []
    precision = []
    ndcg = []
    for k in ks:
        hits = hit(corrupt_index, order[:k])
        recall.append(recallk(corrupt_index, hits))
        precision.append(precisionk(order[:k], hits))
        ndcg.append(ndcgk(corrupt_index, order[:k], hits))

    return recall, precision, ndcg