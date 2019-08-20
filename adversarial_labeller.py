from sklearn.base import TransformerMixin, LinearClassifierMixin

class AdversarialLabeller(TransformerMixin, LinearClassifierMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return self.predict_prob(X)

#todo: make sure this works as expected:
# 
# can instantiate an AdversarialLabeller with, say,
# a randomforest, use fit with appropriately transformed data
# then can use transform and get probabilities, would be pretty cool
# if this simple