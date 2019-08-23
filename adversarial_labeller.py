from sklearn.base import TransformerMixin
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

class AdversarialRandomForestLabeller(RandomForestClassifier, TransformerMixin):
    def __init__(self, args_for_randomforest):
        super().__init__(**args_for_randomforest)

    def transform(self, X):
        return self.predict_proba(X)

class AdversarialExtraTreesLabeller(ExtraTreesClassifier, TransformerMixin):
    def __init__(self, args_for_classifer):
        super().__init__(**args_for_classifer)

    def transform(self, X):
        return self.predict_proba(X)

class AdversarialLogisticRegressionLabeller(LogisticRegression, TransformerMixin):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        return self.predict_proba(X)