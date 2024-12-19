# small perturbation of input and we want to see if the original image could be selected as top explanation.
import numpy as np
import torch

def perturbation_explanation(explainer, dataloader, size=100, seed=1, topk=3, flip=False, gpu=False):

    influences = []

    data_size = len(dataloader.dataset)
    np.random.seed(seed)
    data_indexes = np.random.choice(data_size, size)

    data_features = []
    for i in data_indexes:
        feature, label = dataloader.dataset[i]
        data_features.append(feature)

    X = np.stack(data_features)
    Xtensor = torch.from_numpy(X)
    if flip:
        Xtensor = torch.flip(Xtensor, (3,))
        Xperturbed = Xtensor
    else:

        G = torch.Generator()
        G.manual_seed(seed)

        var = torch.var(Xtensor) * 0.01

        perturbation = torch.FloatTensor(Xtensor.shape).uniform_(-var, var, generator=G)
        Xperturbed = torch.clip(Xtensor + perturbation, torch.min(Xtensor), torch.max(Xtensor))

    Xperturbed = Xperturbed.detach()

    if gpu:
        Xperturbed = Xperturbed.cuda()

    _, influence_scores = explainer.pred_explanation(dataloader, Xperturbed, topK=3)
    influences.append(influence_scores)

    full_influence_matrix = np.vstack(influences)

    topk_influencers = np.argpartition(full_influence_matrix, -topk, axis=1)[:, -topk:]
    ideal_influencer = data_indexes.reshape(-1,1)

    AA = topk_influencers.reshape(-1,3,1)
    BB = ideal_influencer.reshape(-1,1,1)

    return (len(np.unique(topk_influencers.flatten()))/(topk * size), 
            (AA == BB).sum(-1).astype(bool).sum()/topk_influencers.shape[0])



def perturbation(original_dataloader, size=100, seed=1, flip=False, gpu=False):
    # importing the dataloader
    from torch.utils.data import DataLoader, TensorDataset

    data_size = len(original_dataloader.dataset)
    np.random.seed(seed)
    data_indexes = np.random.choice(data_size, size)

    data_features = []
    data_labels = []
    for i in data_indexes:
        feature, label = original_dataloader.dataset[i]
        data_features.append(feature)
        data_labels.append(label)

    X = np.stack(data_features)
    y = torch.tensor(data_labels)

    Xtensor = torch.from_numpy(X)
    if flip:
        Xtensor = torch.flip(Xtensor, (3,))
        Xperturbed = Xtensor
    else:
        G = torch.Generator()
        G.manual_seed(seed)

        var = torch.var(Xtensor) * 0.01
        perturbation = torch.FloatTensor(Xtensor.shape).uniform_(-var, var, generator=G)
        Xperturbed = torch.clip(Xtensor + perturbation, torch.min(Xtensor), torch.max(Xtensor))

    Xperturbed = Xperturbed.detach()
    if gpu:
        Xperturbed = Xperturbed.cuda()
        y = y.cuda()

    # Create a new dataset and dataloader
    perturbed_dataset = TensorDataset(Xperturbed, y)
    perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=original_dataloader.batch_size, shuffle=True)

    return perturbed_dataloader, data_indexes

    