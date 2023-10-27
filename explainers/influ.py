import torch
import torch.nn.functional as F

from torch.autograd import grad
from utils.progress import display_progress

import numpy as np
from tqdm import tqdm

from explainers import BaseExplainer


class InfluenceFunction(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False, scale=False, **kwargs):
        super(InfluenceFunction,self).__init__(classifier, n_classes, gpu)
        self.last_layer = False
        if scale:
            self.last_layer =True

    def data_influence(self, train_loader, cache=True, **kwargs):
        
        grad_zs = []
        for i, data in enumerate(tqdm(train_loader)):
            Xtensor, ytensor = data
            for i in range(Xtensor.shape[0]):
                grad_z_vec = self.grad_z(Xtensor[i:i+1], ytensor[i:i+1])
                grad_zs.append(grad_z_vec)

        if cache == True:
            self.influence = grad_zs
        else:
            return grad_zs
        
    def pred_explanation(self, train_loader, X_test, topK=5):
        X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        if self.gpu:
            X_test_tensor = X_test_tensor.cuda()
        y_test_hat = torch.argmax(self.classifier.predict(X_test_tensor), dim=1).detach()

        s_test_vec = self.calc_s_test(X_test_tensor, y_test_hat, train_loader)

        scores = np.array([self.calc_influence_function(s_test_vec[i]) for i in range(len(s_test_vec))])

        return np.argpartition(scores, -topK, axis=1)[:, -topK:], scores
    
    def data_debugging(self, train_loader):

        s_test_vec = []
        for Xtensor, _ in train_loader:
            if self.gpu:
                Xtensor = Xtensor.cuda()
            y_pred_tensor = torch.argmax(self.classifier.predict(Xtensor), dim=1).detach()

            s_test_vec.extend(self.calc_s_test(Xtensor, y_pred_tensor, train_loader))

        scores = np.array([self.calc_self_influence_function(s_test_vec[i], i) for i in range(len(s_test_vec))])

        return -scores, np.argsort(scores)[::-1]


    def grad_z(self, z, t):

        self.classifier.eval()

        # initialize
        if self.gpu:
            z, t = z.cuda(), t.cuda()

        y = self.classifier.predict(z)
        loss = self.calc_loss(y, t)
        # Compute sum of gradients from model parameters to loss
        if self.last_layer:
            params = [ p for p in self.classifier.fc.parameters() if p.requires_grad ]
        else:
            params = [ p for p in self.classifier.parameters() if p.requires_grad ]
        return list(grad(loss, params))
    
    @staticmethod
    def calc_loss(y, t):
        loss = torch.nn.functional.nll_loss(
            y, t, weight=None, reduction='mean')
        return loss
    
    def calc_s_test(self,  X_test, y_test, train_loader,
                damp=0.01, scale=25, recursion_depth=10, r=1):

        s_tests = []
        for i in tqdm(range(X_test.shape[0])):
            s_test_vec = self.calc_s_test_single(X_test[i:i+1], y_test[i:i+1], train_loader,
                                                 damp, scale, recursion_depth, r)

            s_tests.append(s_test_vec)

        return s_tests
    
    def calc_s_test_single(self, z_test, t_test, train_loader,
                       damp=0.01, scale=25, recursion_depth=5, r=1):
        
        all = self.s_test(z_test, t_test, train_loader, damp=damp, scale=scale,
                 recursion_depth=recursion_depth)
        for i in range(1, r):
            cur = self.s_test(z_test, t_test, train_loader, damp=damp, scale=scale,
            recursion_depth=recursion_depth)
            all = [a + c for a, c in zip(all, cur)]

        s_test_vec = [a / r for a in all]

        return s_test_vec

    
    def s_test(self, z_test, t_test, train_loader, damp=0.01, scale=25.0,
           recursion_depth=5):
        v = self.grad_z(z_test, t_test)
        h_estimate = v.copy()

        for j in range(recursion_depth):
            for Xtensor, ytensor in train_loader:
                if self.gpu:
                    Xtensor, ytensor = Xtensor.cuda(), ytensor.cuda()
                y_hat = self.classifier.predict(Xtensor)
                loss = self.calc_loss(y_hat, ytensor)
                if self.last_layer:
                    params = [ p for p in self.classifier.fc.parameters() if p.requires_grad ]
                else:
                    params = [ p for p in self.classifier.parameters() if p.requires_grad ]
                hv = self.hvp(loss, params, h_estimate)
                # Recursively caclulate h_estimate
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                break
        return h_estimate
    
    @staticmethod
    def hvp(y, w, v):
        if len(w) != len(v):
            raise(ValueError("w and v must have the same length."))

        # First backprop
        first_grads = grad(y, w, retain_graph=True, create_graph=True)

        # Elementwise products
        elemwise_products = 0
        for grad_elem, v_elem in zip(first_grads, v):
            elemwise_products += torch.sum(grad_elem * v_elem)

        # Second backprop
        return_grads = grad(elemwise_products, w)

        return return_grads
    
    def calc_influence_function(self, e_s_test):

        train_dataset_size = len(self.influence)
        influences = []
        for i in tqdm(range(train_dataset_size)):
            tmp_influence = sum(
                [
                    ###################################
                    # TODO: verify if computation really needs to be done
                    # on the CPU or if GPU would work, too
                    ###################################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(self.influence[i], e_s_test)
                    ###################################
                    # Originally with [i] because each grad_z contained
                    # a list of tensors as long as e_s_test list
                    # There is one grad_z per training data sample
                    ###################################
                ]) / train_dataset_size
            influences.append(tmp_influence)

        return influences
    
    def calc_self_influence_function(self, e_s_test, i):

        train_dataset_size = len(self.influence)

        influence = sum(
            [
                ###################################
                # TODO: verify if computation really needs to be done
                # on the CPU or if GPU would work, too
                ###################################
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(self.influence[i], e_s_test)
                ###################################
                # Originally with [i] because each grad_z contained
                # a list of tensors as long as e_s_test list
                # There is one grad_z per training data sample
                ###################################
            ]) 

        return influence