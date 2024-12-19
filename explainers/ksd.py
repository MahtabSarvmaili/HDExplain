import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from explainers import BaseExplainer


class KSDExplainer(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False, scale=False, kernel="RBF", temperature= 1.0, **kwargs):
        super(KSDExplainer,self).__init__(classifier, n_classes, gpu)
        self.last_layer = False
        if scale:
            self.last_layer =True
        if kernel == "RBF":
            self.kernel = self.gaussian_stein_kernel
        elif kernel == "IMQ":
            self.kernel = self.imq_stein_kernel
        else:
            self.kernel = self.liner_stein_kernel
        self.temperature = temperature
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

    def data_influence(self, train_loader, cache=True, **kwargs):

        DXY = []
        for i, data in enumerate(tqdm(train_loader)):
            Xtensor, ytensor = data
            if self.gpu:
                Xtensor = Xtensor.to(self.device)
                ytensor = ytensor.to(self.device)
            yonehot = F.one_hot(ytensor, num_classes=self.n_classes)
            # print(yonehot)
            xbackpropable = Xtensor.clone().detach()
            if not self.last_layer:
                xbackpropable.requires_grad = True
                pred = self.classifier.predict(xbackpropable)
                pred_prob = torch.sum(pred * yonehot, dim=1)
                log_pred_prob = torch.log(pred_prob)
                output = torch.sum(log_pred_prob)
                gradients = torch.autograd.grad(output, xbackpropable)[0]
            else:
                representation = self.classifier.representation(xbackpropable)
                pred = self.classifier.predict_with_representation(representation)
                pred_prob = torch.sum(pred * yonehot, dim=1)
                log_pred_prob = torch.log(pred_prob)
                output = torch.sum(log_pred_prob)
                gradients = torch.autograd.grad(output, representation)[0]

            DXY.append(np.hstack([self.to_np(gradients.reshape(gradients.shape[0], -1)),
                            self.to_np(pred)]))

        self.influence = np.vstack(DXY)
        
    def _data_influence(self, X, y):

        yonehot = F.one_hot(y.clone().detach(), num_classes=self.n_classes)
        # print(yonehot)
        xbackpropable = X.clone().detach()
        if not self.last_layer:
            xbackpropable.requires_grad = True
            pred = self.classifier.predict(xbackpropable)
            pred_prob = torch.sum(pred * yonehot, dim=1)
            log_pred_prob = torch.log(pred_prob)
            output = torch.sum(log_pred_prob)
            gradients = torch.autograd.grad(output, xbackpropable)[0]
        else:
            representation = self.classifier.representation(xbackpropable)
            pred = self.classifier.predict_with_representation(representation)
            pred_prob = torch.sum(pred * yonehot, dim=1)
            log_pred_prob = torch.log(pred_prob)
            output = torch.sum(log_pred_prob)
            gradients = torch.autograd.grad(output, representation)[0]

        DXY = np.hstack([self.to_np(gradients.reshape(gradients.shape[0], -1)),
                        self.to_np(pred)])
        
        return DXY, self.to_np(yonehot)

    @staticmethod
    def gaussian_stein_kernel(
        x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, 
        sigma, return_kernel=False, **kwargs
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

    @staticmethod
    def ksd_debug_kernel(
            x: np.ndarray, y: np.ndarray,
            scores_x: np.ndarray, scores_y: np.ndarray,
            pred_prob_x: np.ndarray, pred_prob_y: np.ndarray,
            sigma: float = 1.
    ):
        _, p = x.shape
        d = x[:, None, :] - y[None, :, :]
        dists = (d ** 2).sum(axis=-1)

        k = np.exp(-dists / sigma / 2)
        scalars = np.matmul(scores_x, scores_y.T)
        scores_diff = scores_x[:, None, :] - scores_y[None, :, :]
        diffs = (d * scores_diff).sum(axis=-1)

        der2 = p - dists / sigma
        stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
        weights = np.outer(pred_prob_x, pred_prob_y)
        weighted_stein_kernel = stein_kernel * weights
        scores_diff = scores_diff.sum(axis=-1)
        return k, scalars, scores_diff, diffs, weighted_stein_kernel

    @staticmethod
    def liner_stein_kernel(
        x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, 
        return_kernel=False, **kwargs):

        n, d = x.shape
        kernel = np.matmul(x, y.T)
        stein_kernel = (
            np.matmul(scores_x, scores_y.T) * kernel + np.matmul(scores_x, y.T) + np.matmul(scores_y, x.T).T + d
        )

        weights = np.outer(pred_prob_x, pred_prob_y)
        weighted_stein_kernel = stein_kernel * weights

        if return_kernel:
            return weighted_stein_kernel, kernel
        return weighted_stein_kernel
    
    @staticmethod
    def imq_stein_kernel(x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, 
                         g=1, beta=0.5, return_kernel=False, **kwargs):
        _, p = x.shape
        d = x[:, None, :] - y[None, :, :]
        dists = (d ** 2).sum(axis=-1)
        res = 1 + g * dists
        kxy = res ** (-beta)
        scores_d = scores_x[:, None, :] - scores_y[None, :, :]
        temp = d * scores_d
        dkxy = 2 * beta * g * (res) ** (-beta - 1) * temp.sum(axis=-1)
        d2kxy = 2 * (
            beta * g * (res) ** (-beta - 1) * p
            - 2 * beta * (beta + 1) * g ** 2 * dists * res ** (-beta - 2)
        )
        k_pi = np.matmul(scores_x,scores_y.T) * kxy + dkxy + d2kxy
        weights = np.outer(pred_prob_x, pred_prob_y)
        weighted_stein_kernel = k_pi * weights

        if return_kernel:
            return weighted_stein_kernel, kxy
        return weighted_stein_kernel
    
    @staticmethod
    def elementwise_gaussian_stein_kernel(
        x, y, scores_x, scores_y, pred_prob_x, pred_prob_y, 
        sigma, return_kernel=False
    ):
        _, p = x.shape
        d = x - y
        dists = (d ** 2).sum(axis=-1)
        k = np.exp(-dists / sigma / 2)
        scalars = (scores_x * scores_y).sum(axis=-1)
        scores_diffs = scores_x - scores_y
        # print(f'score diffs: {scores_diffs.shape}')
        diffs = (d * scores_diffs).sum(axis=-1)
        der2 = p - dists / sigma
        stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
        weights = pred_prob_x * pred_prob_y
        weighted_stein_kernel = stein_kernel * weights
        if return_kernel:
            return weighted_stein_kernel, k
        return weighted_stein_kernel


    def data_model_discrepancy(self, X, y):
        yonehot = self.to_np(F.one_hot(torch.tensor(y), num_classes=self.n_classes))
        if not self.last_layer:
            D = np.hstack([X.reshape(X.shape[0], -1),yonehot])
        else:
            Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
            representation = self.classifier.representation(Xtensor)
            D = np.hstack([self.to_np(representation),yonehot])

        DXY = self.influence
        pyx = np.ones(X.shape[0])
        unnormalized = np.sum(self.kernel(D, D, 
                                          DXY, DXY, 
                                          pyx, pyx, 
                                          sigma = self.temperature * D.shape[1]))
        return unnormalized / (X.shape[0] ** 2)


    def inference_transfer(self, X):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        if self.gpu:
            Xtensor = Xtensor.to(self.device)
        y_hat = torch.argmax(self.classifier.predict(Xtensor), dim=1)
        return self._data_influence(Xtensor, y_hat)
    
    def pred_explanation(self, train_loader, X_test, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)

        if not self.last_layer:
            D_test = np.hstack([X_test.reshape(X_test.shape[0], -1),yonehot_test])
        else:
            Xtensor_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
            if self.gpu:
                Xtensor_test = Xtensor_test.to(self.device)
            representation = self.classifier.representation(Xtensor_test)
            D_test = np.hstack([self.to_np(representation),yonehot_test])

        ksd = []
        data_index = 0
        for X,y in tqdm(train_loader):
            yonehot = self.to_np(F.one_hot(y, num_classes=self.n_classes))
            if not self.last_layer:
                D = np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot])
            else:
                if self.gpu:
                    X = X.to(self.device)
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])

            DXY = self.influence[data_index: data_index+yonehot.shape[0]]
            ksd.append(self.kernel(D_test, D, DXY_test, DXY,
                                   1, 1, sigma = self.temperature * D_test.shape[1]))
            data_index = data_index + yonehot.shape[0]

        ksd = np.hstack(ksd)

        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd

    def ksd_debug(self, train_loader, X_test, y_test):

        Xtensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        ytensor = torch.tensor(y_test)
        if self.gpu:
            Xtensor = Xtensor.to(self.device)
        yonehot_test = F.one_hot(ytensor, num_classes=self.n_classes)
        DXY_test, _ = self._data_influence(Xtensor, ytensor)
        if not self.last_layer:
            D_test = np.hstack([X_test.reshape(X_test.shape[0], -1), yonehot_test])
        else:
            Xtensor_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
            if self.gpu:
                Xtensor_test = Xtensor_test.to(self.device)
            representation = self.classifier.representation(Xtensor_test)
            D_test = np.hstack([self.to_np(representation),yonehot_test])

        ksd = []
        data_index = 0
        for X,y in tqdm(train_loader):
            yonehot = self.to_np(F.one_hot(y, num_classes=self.n_classes))
            if not self.last_layer:
                D = np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot])
            else:
                if self.gpu:
                    X = X.to(self.device)
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])

            DXY = self.influence[data_index: data_index+yonehot.shape[0]]
            ksd.append(self.kernel(D_test, D, DXY_test, DXY,
                                   1, 1, sigma = self.temperature * D_test.shape[1]))
            data_index = data_index + yonehot.shape[0]

        ksd = np.hstack(ksd)

        return ksd

    def datapoint_model_discrepancy(self, train_loader, X_test, y_test, dt_i, dt_ids):
        ksd = np.zeros((1, len(dt_ids)))
        rbf_k = np.zeros((1, len(dt_ids)))
        scalars = np.zeros((1, len(dt_ids)))
        scores_diff = np.zeros((1, len(dt_ids)))
        diffs = np.zeros((1, len(dt_ids)))

        Xtensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        ytensor = torch.tensor(y_test)
        if self.gpu:
            Xtensor = Xtensor.to(self.device)
        yonehot_test = F.one_hot(ytensor, num_classes=self.n_classes)

        D = np.hstack([X_test.reshape(X_test.shape[0], -1), yonehot_test])
        DXY = self.influence[dt_i, :].reshape(1, -1)
        pxy = np.ones(D.shape[0])

        index_ = 0
        for j in dt_ids:
            x_, y_ = train_loader.dataset.__getitem__(j)
            x_ = x_.unsqueeze(dim=0).cpu()
            y_ = np.array(y_).reshape(1)
            y_ = torch.tensor(y_)

            y_onehot_ = self.to_np(F.one_hot(y_, num_classes=self.n_classes))

            D_ = np.hstack([self.to_np(x_.reshape(x_.shape[0], -1)), y_onehot_])
            DXY_ = self.influence[j, :].reshape(1, -1)
            pxy_ = np.ones(D_.shape[0])
            rbf_k[:, index_], scalars[:, index_], scores_diff[:, index_], diffs[:index_], ksd[:, index_] = self.ksd_debug_kernel(
                D, D_, DXY, DXY_, pxy, pxy_, D_.shape[1] - self.n_classes)

            index_ += x_.shape[0]
        res = {
            "rbf_kernel": rbf_k.reshape(-1),
            "scalars": scalars.reshape(-1),
            "scores_diff": np.abs(scores_diff.reshape(-1)),
            "ksd": ksd.reshape(-1)
        }
        return res

    def pred_explanation_with_cropping(self, train_loader, X_test, boxes, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)

        if not self.last_layer:
            D_test = np.hstack([X_test.reshape(X_test.shape[0], -1),yonehot_test])
        else:
            Xtensor_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
            if self.gpu:
                Xtensor_test = Xtensor_test.to(self.device)
            representation = self.classifier.representation(Xtensor_test)
            D_test = np.hstack([self.to_np(representation),yonehot_test])

        ksd = []
        ksd_cropped = []
        data_index = 0
        for X,y in tqdm(train_loader):
            yonehot = self.to_np(F.one_hot(y, num_classes=self.n_classes))
            if not self.last_layer:
                D = np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot])
            else:
                if self.gpu:
                    X = X.to(self.device)
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])

            DXY = self.influence[data_index: data_index+yonehot.shape[0]]
            ksd.append(self.kernel(D_test, D, DXY_test, DXY,
                                   1, 1, sigma = self.temperature * D_test.shape[1]))

            partial_ksd = []
            for i, box in enumerate(boxes):
                D_test_ins = D_test[i:i+1]
                DXY_test_ins = DXY_test[i:i+1]

                cropped_D_test_inst = self.crop(D_test_ins, X_test.shape, box)

                partial_ksd.append(self.kernel(cropped_D_test_inst, 
                                               self.crop(D, X_test.shape, box), 
                                               self.crop(DXY_test_ins, X.shape, box), 
                                               self.crop(DXY, X.shape, box), 
                                               1, 1, sigma = self.temperature * cropped_D_test_inst.shape[1]))
            ksd_cropped.append(np.vstack(partial_ksd))
            data_index = data_index + yonehot.shape[0]

        ksd = np.hstack(ksd)
        ksd_cropped = np.hstack(ksd_cropped)

        ksd = ksd + ksd_cropped

        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd

    def crop(self, array, input_shape, box):
        feature_dim = np.prod(input_shape[1:])
        X = array[:,:feature_dim].reshape((array.shape[0],)+input_shape[1:])
        y = array[:,feature_dim:]
        cropped_X = X[:, :, box[0]:box[1], box[2]:box[3]]
        return np.hstack([cropped_X.reshape(array.shape[0], -1), y])
  
    def data_debugging(self, train_loader):
        ksd = []
        index = 0
        for i, (X,y) in enumerate(tqdm(train_loader)):
            yonehot = self.to_np(F.one_hot(y, num_classes=self.n_classes))

            if not self.last_layer:
                D = np.hstack([self.to_np(X.reshape(X.shape[0], -1)),yonehot])
            else:
                if self.gpu:
                    X = X.to(self.device)
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])   

            DXY = self.influence[index:index+X.shape[0]]

            ksd.append(self.elementwise_gaussian_stein_kernel(D, D, DXY, DXY, 
                                                  1, 1, self.temperature * D.shape[1]))
            index += X.shape[0]
        
        ksd = np.concatenate(ksd)
        
        return ksd, np.argsort(ksd)[::-1]







