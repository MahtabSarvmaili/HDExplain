import torch
import torch.nn.functional as F

import numpy as np

from explainers import BaseExplainer


class KSDExplainer(BaseExplainer):
    def __init__(self, classifier, n_classes):
        super(KSDExplainer,self).__init__(classifier, n_classes)

    def data_influence(self, X, y, cache=True):
        yonehot = F.one_hot(torch.tensor(y), num_classes=self.n_classes)
        # print(yonehot)
        xbackpropable = torch.from_numpy(np.array(X, dtype=np.float32))
        xbackpropable.requires_grad = True
        pred = self.classifier.predict(xbackpropable)
        pred_prob = torch.sum(pred * yonehot, dim=1)
        log_pred_prob = torch.log(pred_prob)
        output = torch.sum(log_pred_prob)
        gradients = torch.autograd.grad(output, xbackpropable)[0]

        DXY = np.hstack([gradients.detach().numpy(),
                        pred.detach().numpy()])

        if cache == True:
            self.influence = (DXY, 
                            yonehot.detach().numpy()
                            )
        else:
            return (DXY, 
                    yonehot.detach().numpy()
                    )

    @staticmethod
    def gaussian_stein_kernel(
        x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, 
        sigma, return_kernel=False
    ):
        _, p = x.shape
        d = x[:, None, :] - y[None, :, :]
        dists = (d ** 2).sum(axis=-1)
        k = np.exp(-dists / sigma / 2)
        scalars = np.matmul(scores_x, scores_y.T)
        scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
        # print(f'score diffs: {scores_diffs.shape}')
        diffs = (d * scores_diffs).sum(axis=-1)
        der2 = p - dists / sigma
        stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
        weights = np.outer(pred_prob_x, pred_prob_y)
        weighted_stein_kernel = stein_kernel * weights
        if return_kernel:
            return weighted_stein_kernel, k
        return weighted_stein_kernel


    def data_model_discrepancy(self, X):
        D = np.hstack([X,self.influence[1]])
        DXY = self.influence[0]
        pyx = np.ones(X.shape[0])
        unnormalized = np.sum(self.gaussian_stein_kernel(D, D, 
                                                         DXY, DXY, 
                                                         pyx, pyx, 
                                                         1))
        return unnormalized / (X.shape[0] ** 2)


    def inference_transfer(self, X):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        y_hat = torch.argmax(self.classifier.predict(Xtensor), dim=1).detach()
        return self.data_influence(X, y_hat, cache=False)
    
    def pred_explanation(self, X, X_test, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)
        D_test = np.hstack([X_test,yonehot_test])
        D = np.hstack([X,self.influence[1]])
        DXY = self.influence[0]
        ksd = self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 1, 1, 1)
        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd
