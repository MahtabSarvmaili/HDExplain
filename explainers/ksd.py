import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from explainers import BaseExplainer


class KSDExplainer(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False, scale=False, **kwargs):
        super(KSDExplainer,self).__init__(classifier, n_classes, gpu)
        self.last_layer = False
        if scale:
            self.last_layer =True

    def data_influence(self, train_loader, cache=True, **kwargs):

        DXY = []
        for i, data in enumerate(tqdm(train_loader)):
            Xtensor, ytensor = data
            if self.gpu:
                Xtensor = Xtensor.cuda()
                ytensor = ytensor.cuda()
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
        unnormalized = np.sum(self.gaussian_stein_kernel(D, D, 
                                                         DXY, DXY, 
                                                         pyx, pyx, 
                                                         D.shape[1]))
        return unnormalized / (X.shape[0] ** 2)


    def inference_transfer(self, X):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        if self.gpu:
            Xtensor = Xtensor.cuda()
        y_hat = torch.argmax(self.classifier.predict(Xtensor), dim=1)
        return self._data_influence(Xtensor, y_hat)
    
    def pred_explanation(self, train_loader, X_test, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)

        if not self.last_layer:
            D_test = np.hstack([X_test.reshape(X_test.shape[0], -1),yonehot_test])
        else:
            Xtensor_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
            if self.gpu:
                Xtensor_test = Xtensor_test.cuda()
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
                    X = X.cuda()
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])   

            DXY = self.influence[data_index: data_index+yonehot.shape[0]]
            ksd.append(self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 
                                            1, 1, D_test.shape[1]))
            data_index = data_index + yonehot.shape[0]

        ksd = np.hstack(ksd)

        return np.argpartition(ksd, -topK, axis=1)[:, -topK:], ksd

    def pred_explanation_with_cropping(self, train_loader, X_test, boxes, topK=5):
        DXY_test, yonehot_test = self.inference_transfer(X_test)

        if not self.last_layer:
            D_test = np.hstack([X_test.reshape(X_test.shape[0], -1),yonehot_test])
        else:
            Xtensor_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
            if self.gpu:
                Xtensor_test = Xtensor_test.cuda()
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
                    X = X.cuda()
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])   

            DXY = self.influence[data_index: data_index+yonehot.shape[0]]
            ksd.append(self.gaussian_stein_kernel(D_test, D, DXY_test, DXY, 
                                            1, 1, D_test.shape[1]))
            
            partial_ksd = []
            for i, box in enumerate(boxes):
                D_test_ins = D_test[i:i+1]
                DXY_test_ins = DXY_test[i:i+1]

                cropped_D_test_inst = self.crop(D_test_ins, X_test.shape, box)

                partial_ksd.append(self.gaussian_stein_kernel(cropped_D_test_inst, 
                                                            self.crop(D, X_test.shape, box), 
                                                            self.crop(DXY_test_ins, X.shape, box), 
                                                            self.crop(DXY, X.shape, box), 
                                                1, 1, cropped_D_test_inst.shape[1]))
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
                    X = X.cuda()
                representation = self.classifier.representation(X)
                D = np.hstack([self.to_np(representation),yonehot])   

            DXY = self.influence[index:index+X.shape[0]]

            ksd.append(self.elementwise_gaussian_stein_kernel(D, D, DXY, DXY, 
                                                  1, 1, D.shape[1]))
            index += X.shape[0]
        
        ksd = np.concatenate(ksd)
        
        return ksd, np.argsort(ksd)
