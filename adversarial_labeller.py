from sklearn.base import TransformerMixin
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class AdversarialRandomForestLabeller(RandomForestClassifier, TransformerMixin):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        return self.predict_proba(X)

class AdversarialLogisticRegressionLabeller(LogisticRegression, TransformerMixin):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        return self.predict_proba(X)