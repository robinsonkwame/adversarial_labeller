import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


class AdversarialRandomForestLabeller(RandomForestClassifier, TransformerMixin):
    def __init__(self, fit_params=None):
        super().__init__(**fit_params)
        self.params = fit_params

    def transform(self, X):
        return self.predict_proba(X)
    
    def get_params(self, deep=False):
        return self.params

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

class AdversarialLogisticRegressionCVLabeller(LogisticRegressionCV, TransformerMixin):
    def __init__(self, random_state, cv, solver, class_weight, penalty):
        super().__init__(
            penalty=penalty,
            class_weight=class_weight,
            random_state=random_state,
            cv=cv,
            solver=solver)

    def transform(self, X):
        return self.predict_proba(X)

    def get_params(self, deep=False):
        return {
            'random-state': self.random_state,
            'cv': self.cv
        }

class RandomForestRandomizedCV:
    # from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    class_weight=[None, "balanced", "balanced_subsample"]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': class_weight}
    
    def get_best_parameters(self,
                            features,
                            labels,
                            n_iter=100,
                            cv=3,
                            verbose=2,
                            random_state=1,
                            n_jobs=-1):
        rf_random =\
            RandomizedSearchCV(
                estimator=RandomForestClassifier(),
                param_distributions=self.random_grid,
                n_iter=n_iter,
                cv=cv,
                verbose=verbose,
                random_state=random_state,
                n_jobs=n_jobs
            )
        rf_random.fit(
            features,
            labels
        )

        return rf_random.best_params_