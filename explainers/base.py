from abc import ABC, abstractmethod


class BaseExplainer(object):
    def __init__(self, classifier, n_classes):
        self.classifier = classifier
        self.n_classes = n_classes
        self.influence = None
        
        # set model to eval mode
        self.classifier.eval()

    @abstractmethod   
    def data_influence(self, X, y, cache=True):
        pass

    @abstractmethod
    def pred_explanation(self, X, X_test, topK=5):
        pass