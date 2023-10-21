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
    
    def s_test(self, z_test, t_test, z_loader, damp=0.01, scale=25.0,
           recursion_depth=5000):
        
        v = self.grad_z(z_test, t_test)
        h_estimate = v.copy()

        for i in range(recursion_depth):
            for x, t in z_loader:
                if self.gpu:
                    x, t = x.cuda(), t.cuda()
                y = self.classifier.model(x)
                loss = self.calc_loss(y, t)
                params = [ p for p in self.classifier.parameters() if p.requires_grad ]
                hv = self.hvp(loss, params, h_estimate)
                # Recursively caclulate h_estimate
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                break
            display_progress("Calc. s_test recursions: ", i, recursion_depth)
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