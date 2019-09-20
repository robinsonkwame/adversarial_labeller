from sklearn.metrics import accuracy_score

class Scorer:
    def __init__(self, the_scorer, a_pipeline=None):
        self.the_scorer = the_scorer
        self.a_pipeline = a_pipeline

    def grade(self, estimator, X, y):
        X_ = X
        if self.a_pipeline:
            X_ = self.a_pipeline.transform(X)

        labelled_test_mask =\
            self.the_scorer.predict(
                X.values
            ) == 1

        return  accuracy_score(y_true= y[labelled_test_mask],
                            y_pred= estimator.predict(
                                        X_[labelled_test_mask].values
                                        )
                )
