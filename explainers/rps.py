import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from explainers import BaseExplainer

class Sigmoid(nn.Module):
    def __init__(self, W):
        super(Sigmoid, self).__init__()
        self.W = Variable(W, requires_grad=True)

    def forward(self, x):
        # calculate output and L2 regularizer
        H = torch.matmul(x, self.W.transpose(0, 1))
        Phi = F.sigmoid(H)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return Phi, L2
    

class RepresenterPointSelection(BaseExplainer):
    def __init__(self, classifier, n_classes, gpu=False):
        super(RepresenterPointSelection,self).__init__(classifier, n_classes, gpu)
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            self.model = Sigmoid(classifier.fc.weight.data.cpu().detach()).cuda()
        else:
            self.dtype = torch.FloatTensor
            self.model = Sigmoid(classifier.fc.weight.data.detach())


    def data_influence(self, X, y, cache=True, lmbd=0.003, epoch=3000, **kwargs):
        Xtensor = torch.from_numpy(np.array(X, dtype=np.float32))
        if self.gpu:
            Xtensor = Xtensor.cuda()
        Xrepresentation = self.classifier.representation(Xtensor).data.detach()
        pred = self.classifier.predict(Xtensor).data.detach()
        if self.gpu:
            Xrepresentation = Xrepresentation.cuda()
            pred = pred.cuda()

        if cache == True:
            alpha = self.retrain(Xrepresentation, pred, self.model, lmbd, epoch)
            self.influence = (alpha, self.to_np(Xrepresentation))
        else:
            pred_label = self.to_np(F.one_hot(torch.argmax(pred, dim=1)))
            return pred_label, self.to_np(Xrepresentation)
        
    def pred_explanation(self, X, y, X_test, topK=5):
        test_pred_label, test_representation = self.data_influence(X_test, None, cache=False)
        alpha, train_representation = self.influence
        alpha_j = np.matmul(alpha, test_pred_label.T)

        representation_similarity = np.matmul(train_representation, test_representation.T)

        scores = (representation_similarity * alpha_j).T
        return np.argpartition(scores, -topK, axis=1)[:, -topK:], scores 

    def data_debugging(self, X, y):
        alpha, _ = self.influence

        alpha_j = alpha[range(alpha.shape[0]), y]

        return np.sorted(np.diag(alpha_j))[::-1]

    def retrain(self, x, y, model, lmbd, epoch):
        # Fine tune the last layer
        min_loss = 10000.0
        optimizer = optim.SGD([model.W], lr=1.0)
        N = len(y)
        for epoch in range(epoch):
            phi_loss = 0
            optimizer.zero_grad()
            (Phi, L2) = model(x)
            loss = L2 * lmbd + F.binary_cross_entropy(Phi.float(), y.float())
            phi_loss += self.to_np(F.binary_cross_entropy(Phi.float(), y.float()))
            loss.backward()
            temp_W = model.W.data
            grad_loss_W = self.to_np(torch.mean(torch.abs(model.W.grad)))
            # save the W with lowest loss
            if grad_loss_W < min_loss:
                if epoch == 0:
                    init_grad = grad_loss_W
                min_loss = grad_loss_W
                best_W = temp_W
                if min_loss < init_grad / 200:
                    print('stopping criteria reached in epoch :{}'.format(epoch))
                    break
            self.backtracking_line_search(model, model.W.grad, x, y, loss, lambda_l2=lmbd)
            if epoch % 100 == 0:
                print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(epoch, self.to_np(loss), phi_loss, grad_loss_W))
        # caluculate w based on the representer theorem's decomposition
        temp = torch.matmul(x, Variable(best_W).transpose(0, 1))
        sigmoid_value = F.sigmoid(temp)
        # derivative of sigmoid+BCE
        weight_matrix = sigmoid_value - y
        weight_matrix = torch.div(weight_matrix, (-2.0 * lmbd * N))
        return self.to_np(weight_matrix)

    # implmentation for backtracking line search
    def backtracking_line_search(self, model, grad_w, x, y, val, lambda_l2=0.001):
        t = 10.0
        beta = 0.5
        W_O = self.to_np(model.W)
        grad_np_w = self.to_np(grad_w)
        while (True):
            model.W = Variable(torch.from_numpy(W_O - t * grad_np_w).type(self.dtype), requires_grad=True)
            val_n = 0.0
            (Phi, L2) = model(x)
            val_n = F.binary_cross_entropy(Phi.float(), y.float()) + L2 * lambda_l2
            if t < 0.0000000001:
                # print("t too small")
                break
            if self.to_np(val_n - val + t * (torch.norm(grad_w) ** 2) / 2) >= 0:
                t = beta * t
            else:
                break