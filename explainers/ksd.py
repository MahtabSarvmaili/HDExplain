import torch
import torch.nn.functional as F

import numpy as np

from explainers import BaseExplainer


class KSDExplainer(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False):
        super(KSDExplainer,self).__init__(classifier, n_classes, gpu)

    def data_influence(self, train_loader, cache=True, **kwargs):

        DXY = []
        for i, data in enumerate(train_loader):
            Xtensor, ytensor = data
            yonehot = F.one_hot(ytensor, num_classes=self.n_classes)
            # print(yonehot)
            xbackpropable = Xtensor.clone().detach()
            xbackpropable.requires_grad = True
            pred = self.classifier.predict(xbackpropable)
            pred_prob = torch.sum(pred * yonehot, dim=1)
            log_pred_prob = torch.log(pred_prob)
            output = torch.sum(log_pred_prob)
            gradients = torch.autograd.grad(output, xbackpropable)[0]

            DXY.append(np.hstack([self.to_np(gradients),
                            self.to_np(pred)]))

        self.influence = np.vstack(DXY)
        
    def _data_influence(self, X, y):

        yonehot = F.one_hot(y.clone().detach(), num_classes=self.n_classes)
        # print(yonehot)
        xbackpropable = X.clone().detach()
        xbackpropable.requires_grad = True
        pred = self.classifier.predict(xbackpropable)
        pred_prob = torch.sum(pred * yonehot, dim=1)
        log_pred_prob = torch.log(pred_prob)
        output = torch.sum(log_pred_prob)
        gradients = torch.autograd.grad(output, xbackpropable)[0]

        DXY = np.hstack([self.to_np(gradients),
                        self.to_np(pred)])
        
        return DXY, self.to_np(yonehot)

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


    def data_model_discrepancy(self, X, y):
        yonehot = self.to_np(F.one_hot(torch.tensor(y), num_classes=self.n_classes))
        D = np.hstack([X,yonehot])
        DXY = self.influence
        pyx = np.ones(X.shape[0])
        unnormalized = np.sum(self.gaussian_stein_kernel(D, D, 
                                                         DXY, DXY, 
                                                         pyx, pyx, 
                                                         1))
        return unnormalized / (X.shape[0] ** 2)


    def inference_transfer(self, X):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        y_hat = torch.argmax(self.classifier.predict(Xtensor), dim=1)
        return self._data_influence(Xtensor, y_hat)
    
    def pred_explanation(self, X, y, X_test, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)
        yonehot = self.to_np(F.one_hot(torch.tensor(y), num_classes=self.n_classes))
        D_test = np.hstack([X_test,yonehot_test])
        D = np.hstack([X,yonehot])
        DXY = self.influence
        ksd = self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 1, 1, 1)
        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd
    
    def data_debugging(self, X, y):
        DXY_test, yonehot_test = self.inference_transfer(X)
        yonehot = self.to_np(F.one_hot(torch.tensor(y), num_classes=self.n_classes))
        D_test = np.hstack([X,yonehot_test])
        D = np.hstack([X,yonehot])
        DXY = self.influence
        ksd = self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 1, 1, 1)

        return np.sorted(np.diag(ksd))
