"""
Model classes
"""

class ClassifierModel:

    def __init__(self):
        self.model = None
        self.model_name = None
        self.classes = None
        self.num_classes = None

    def train(self, X, Y):
        pass

    def predict(self, X):
        return self.classes[self.model.predict(X)]
