import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


def feature_transfer(net, X, y):
  net.eval()
  yonehot = F.one_hot(torch.tensor(y), num_classes=3)
  print(yonehot)
  xbackpropable = torch.from_numpy(np.array(X, dtype=np.float32))
  xbackpropable.requires_grad = True
  pred = net.predict(xbackpropable)
  pred_prob = torch.sum(pred * yonehot, dim=1)
  log_pred_prob = torch.log(pred_prob)
  output = torch.sum(log_pred_prob)
  gradients = torch.autograd.grad(output, xbackpropable)[0]
  return gradients.detach().numpy(), pred.detach().numpy(), yonehot.detach().numpy()


def gaussian_stein_kernel(
    x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, sigma, return_kernel=False
):
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = np.exp(-dists / sigma / 2)
    scalars = np.matmul(scores_x, scores_y.T)
    scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
    print(f'score diffs: {scores_diffs.shape}')
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    weights = np.outer(pred_prob_x, pred_prob_y)
    weighted_stein_kernel = stein_kernel * weights
    if return_kernel:
        return weighted_stein_kernel, k
    return weighted_stein_kernel


def data_model_discrepancy(X,gX):
  pyx = np.ones(X.shape[0])
  unnormalized = np.sum(gaussian_stein_kernel(X, X, gX, gX, pyx, pyx, 1))
  return unnormalized / (X.shape[0] ** 2)


def inference_transfer(net, X):
  y_hat = torch.argmax(net.predict(torch.from_numpy(np.array(X, dtype=np.float32))), dim=1).detach()
  return feature_transfer(net, X, y_hat)