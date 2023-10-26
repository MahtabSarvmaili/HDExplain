# small perturbation of input and we want to see if the original image could be selected as top explanation.
from tqdm import tqdm
import numpy as np
import torch

def perturbation_explanation(explainer, dataloader, gpu=False):
    influences = []
    for i, data in enumerate(tqdm(dataloader)):
        Xtensor, _ = data 

        G = torch.Generator()
        G.manual_seed(i)

        var = torch.var(Xtensor) * 0.05

        perturbation = torch.FloatTensor(Xtensor.shape).uniform_(-var, var)
        perturbation = 0
        Xperturbed = Xtensor + perturbation

        Xperturbed = Xperturbed.detach()

        if gpu:
            Xperturbed = Xperturbed.cuda()
        _, influence_scores = explainer.pred_explanation(dataloader, Xperturbed, topK=3)
        influences.append(influence_scores)

    full_influence_matrix = np.vstack(influences)
    top_influencer = np.argmax(full_influence_matrix, axis=1)
    ideal_influencer = np.arange(len(top_influencer))
    diff = top_influencer - ideal_influencer

    return np.sum(diff == 0)/len(top_influencer)

