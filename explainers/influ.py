import torch
import torch.nn.functional as F

from torch.autograd import grad
from utils.progress import display_progress

import numpy as np

from explainers import BaseExplainer


class InfluenceFunction(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False):
        super(InfluenceFunction,self).__init__(classifier, n_classes, gpu)

    def data_influence(self, X, y, cache=True):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        ytensor = torch.from_numpy(np.array(y, dtype=np.int_))
        grad_zs = []
        for i in range(Xtensor.shape[0]):
            grad_z_vec = self.grad_z(Xtensor[i:i+1], ytensor[i:i+1])
            grad_zs.append(grad_z_vec)
            # display_progress(
            #     "Calc. grad_z: ", i, Xtensor.shape[0]-i)

        if cache == True:
            self.influence = grad_zs
        else:
            return grad_zs
        
    def pred_explanation(self, X, y, X_test, topK=5):
        X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        y_test_hat = torch.argmax(self.classifier.predict(X_test_tensor), dim=1).detach()

        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        ytensor = torch.from_numpy(np.array(y, dtype=np.int_))

        s_test_vec = self.calc_s_test(X_test_tensor, y_test_hat, Xtensor, ytensor)

        scores = np.array([self.calc_influence_function(s_test_vec[i]) for i in range(len(s_test_vec))])

        return np.argpartition(scores, -topK, axis=1)[:, -topK:], scores


    def grad_z(self, z, t):
        self.classifier.eval()
        # initialize
        if self.gpu:
            z, t = z.cuda(), t.cuda()
        y = self.classifier.predict(z)
        loss = self.calc_loss(y, t)
        # Compute sum of gradients from model parameters to loss
        params = [ p for p in self.classifier.parameters() if p.requires_grad ]
        return list(grad(loss, params, create_graph=True))
    
    @staticmethod
    def calc_loss(y, t):
        loss = torch.nn.functional.nll_loss(
            y, t, weight=None, reduction='mean')
        return loss
    
    def calc_s_test(self,  X_test, y_test, X, y,
                damp=0.01, scale=25, recursion_depth=10, r=1):

        s_tests = []
        for i in range(X_test.shape[0]):
            s_test_vec = self.calc_s_test_single(X_test[i:i+1], y_test[i:i+1], X, y,
                                                 damp, scale, recursion_depth, r)

            s_tests.append(s_test_vec)
            display_progress(
                "Calc. z_test (s_test): ", i, X_test.shape[0])

        return s_tests
    
    def calc_s_test_single(self, z_test, t_test, X, y,
                       damp=0.01, scale=25, recursion_depth=5, r=1):
        
        all = self.s_test(z_test, t_test, X, y, damp=damp, scale=scale,
                 recursion_depth=recursion_depth)
        for i in range(1, r):
            cur = self.s_test(z_test, t_test, X, y, damp=damp, scale=scale,
            recursion_depth=recursion_depth)
            all = [a + c for a, c in zip(all, cur)]
            display_progress("Averaging r-times: ", i, r)    

        s_test_vec = [a / r for a in all]

        return s_test_vec

    
    def s_test(self, z_test, t_test, X, y, damp=0.01, scale=25.0,
           recursion_depth=5):
        v = self.grad_z(z_test, t_test)
        h_estimate = v.copy()

        for j in range(recursion_depth):
            for i in range(X.shape[0]):
                if self.gpu:
                    Xtensor, ytensor = X[i:i+1].cuda(), y[i:i+1].cuda()
                else:
                    Xtensor, ytensor = X[i:i+1], y[i:i+1]
                y_hat = self.classifier.predict(Xtensor)
                loss = self.calc_loss(y_hat, ytensor)
                params = [ p for p in self.classifier.parameters() if p.requires_grad ]
                hv = self.hvp(loss, params, h_estimate)
                # Recursively caclulate h_estimate
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                break
            display_progress("Calc. s_test recursions: ", j, recursion_depth)
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
        return_grads = grad(elemwise_products, w, create_graph=True)

        return return_grads
    
    def calc_influence_function(self, e_s_test):

        train_dataset_size = len(self.influence)
        influences = []
        for i in range(train_dataset_size):
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
            display_progress("Calc. influence function: ", i, train_dataset_size)

        return influences