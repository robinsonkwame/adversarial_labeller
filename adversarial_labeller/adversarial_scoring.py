from sklearn.metrics import accuracy_score

class Scorer:
    def __init__(self, 
                 the_scorer,
                 a_pipeline=None,
                 flip_binary_predictions=False,
                 metric=accuracy_score,
                 predict_proba_threshold=1.0):
        self.metric = metric
        self.the_scorer = the_scorer
        self.a_pipeline = a_pipeline
        self.flip_binary_predictions = flip_binary_predictions
        self.predict_proba_threshold = predict_proba_threshold

    def label(self, X):
        # predictions = self.the_scorer.predict(X)
        # if self.flip_binary_predictions:
        #     predictions ^= 1

        # print("\t predicted proba:" , self.the_scorer.predict_proba(X))
        predicted_proba = self.the_scorer.predict_proba(X)[:,1]
        if self.flip_binary_predictions:
            #predictions ^= 1
            predicted_proba -= 1

        #print("\t predicted proba:" , self.the_scorer.predict_proba(X))
        return predicted_proba

    def grade(self, estimator, X, y):
        score = 0
        _X = X
        if self.a_pipeline:
            _X = self.a_pipeline.transform(X)

        # labelled_test_mask =\
        #     self.label(_X) == 1
        labelled_test_mask =\
            self.label(_X) >= self.predict_proba_threshold 

        #print(labelled_test_mask, 'labelled test mask')

        if any(labelled_test_mask):
            # score = \
            #     accuracy_score(
            #         y_true= y[labelled_test_mask],
            #         y_pred= estimator.predict(
            #                     _X[labelled_test_mask]
            #                 )
            #     )
            score = \
                self.metric(
                    y_true= y[labelled_test_mask],
                    y_pred= estimator.predict(
                                _X[labelled_test_mask]
                            )
                )
        return score