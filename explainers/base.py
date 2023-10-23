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
    def data_influence(self, X, y, cache=True, **kwargs):
        pass

    @abstractmethod
    def pred_explanation(self, X, y, X_test, topK=5):
        pass