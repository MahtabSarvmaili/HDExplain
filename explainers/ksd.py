import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from explainers import BaseExplainer


class KSDExplainer(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False, **kwargs):
        super(KSDExplainer,self).__init__(classifier, n_classes, gpu)

    def data_influence(self, train_loader, cache=True, **kwargs):

        DXY = []
        for data in tqdm(train_loader):
            Xtensor, ytensor = data
            if self.gpu:
                Xtensor = Xtensor.cuda()
                ytensor = ytensor.cuda()
            yonehot = F.one_hot(ytensor, num_classes=self.n_classes)
            # print(yonehot)
            xbackpropable = Xtensor.clone().detach()
            xbackpropable.requires_grad = True
            pred = self.classifier.predict(xbackpropable)
            pred_prob = torch.sum(pred * yonehot, dim=1)
            log_pred_prob = torch.log(pred_prob)
            output = torch.sum(log_pred_prob)
            gradients = torch.autograd.grad(output, xbackpropable)[0]

            DXY.append(np.hstack([self.to_np(gradients.reshape(gradients.shape[0], -1)),
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

        DXY = np.hstack([self.to_np(gradients.reshape(gradients.shape[0], -1)),
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
        D = np.hstack([X.reshape(X.shape[0], -1),yonehot])
        DXY = self.influence
        pyx = np.ones(X.shape[0])
        unnormalized = np.sum(self.gaussian_stein_kernel(D, D, 
                                                         DXY, DXY, 
                                                         pyx, pyx, 
                                                         D.shape[1]-self.n_classes))
        return unnormalized / (X.shape[0] ** 2)


    def inference_transfer(self, X):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        if self.gpu:
            Xtensor = Xtensor.cuda()
        y_hat = torch.argmax(self.classifier.predict(Xtensor), dim=1)
        return self._data_influence(Xtensor, y_hat)
    
    def pred_explanation(self, train_loader, X_test, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)
        D_test = np.hstack([X_test.reshape(X_test.shape[0], -1),yonehot_test])

        D = []
        for X,y in train_loader: 
            yonehot = self.to_np(F.one_hot(y, num_classes=self.n_classes))
            D.append(np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot]))
        D = np.vstack(D)

        DXY = self.influence
        ksd = self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 
                                         1, 1, D_test.shape[1]-self.n_classes)

        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd
    
    def data_debugging(self, train_loader):
        ksd = []
        for X,y in train_loader:
            DXY_test, yonehot_test = self.inference_transfer(X)
            D_test = np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot_test])

            D = []
            for Xt, yt in train_loader:
                yonehot = self.to_np(F.one_hot(yt, num_classes=self.n_classes))
                D = D.append(np.hstack([self.to_np(Xt.reshape(Xt.shape[0], -1)),yonehot]))
            D = np.vstack(D)
            DXY = self.influence
            ksd.append(self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 
                                                  1, 1, D_test.shape[1]-self.n_classes))
        
        ksd = np.vstack(ksd)
        
        return np.sorted(np.diag(ksd))
