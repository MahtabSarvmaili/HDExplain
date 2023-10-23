from abc import ABC, abstractmethod


class BaseExplainer(object):
    def __init__(self, classifier, n_classes, gpu=False):
        self.classifier = classifier
        self.n_classes = n_classes
        self.influence = None
        self.gpu = gpu
        
        # set model to eval mode
        self.classifier.eval()

    @abstractmethod   
    def data_influence(self, train_loader, cache=True, **kwargs):
        pass

    @abstractmethod
    def pred_explanation(self, train_loader, X_test, topK=5):
        pass

    @abstractmethod
    def data_debugging(self, train_loader):
        pass

    def to_np(self, x):
        if self.gpu:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()