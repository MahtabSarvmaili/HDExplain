import torch
import torch.nn.functional as F

from torch.autograd import grad
from utils.progress import display_progress

import numpy as np

from explainers import BaseExplainer


class TracIn(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False):
        super(TracIn,self).__init__(classifier, n_classes, gpu)

    def data_influence(self, X, y, cache=True, **kwargs):
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
    
    def pred_explanation(self, X, y, X_test, topK=5):
        X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
        y_test_hat = torch.argmax(self.classifier.predict(X_test_tensor), dim=1).detach().numpy()
        s_test_vec = self.data_influence(X_test, y_test_hat, cache=False)

        scores = np.array([self.calc_influence_function(s_test_vec[i]) for i in range(len(s_test_vec))])

        return np.argpartition(scores, -topK, axis=1)[:, -topK:], scores

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