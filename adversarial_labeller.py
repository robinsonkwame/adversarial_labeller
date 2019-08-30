import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier

  
class AdversarialRUSBoostLabeller(RUSBoostClassifier, TransformerMixin):
    def __init__(self, fit_params=None):
        super().__init__(**fit_params)
        self.params = fit_params
        self.flip_binary_predictions = False

    def transform(self, X):
        return self.predict_proba(X)
    
    def get_params(self, deep=False):
        return self.params

    def maximize_binary_validation_accuracy(self, X, y):
        validation_score =\
            accuracy_score(
                y,
                self.predict(X)
            )
        if validation_score < 0.50:
            self.flip_binary_predictions = True

    def label(self, X):
        predictions = self.predict(X)
        if self.flip_binary_predictions:
            predictions ^= 1

        return predictions

class AdversarialNearestNeighborLabeller(KNeighborsClassifier, TransformerMixin):
    def __init__(self, fit_params=None):
        super().__init__(**fit_params)
        self.params = fit_params
        self.flip_binary_predictions = False

    def transform(self, X):
        return self.predict_proba(X)
    
    def get_params(self, deep=False):
        return self.params

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

class RUSBoostRandomizedCV:
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)]
    learning_rate = np.linspace(1, 30, 20)
    algorithm=['SAMME.R']  # SAMME will throw ValueError under AdaBoost
    sampling_strategy = ["majority", "not minority", "not majority", "all"]
    replacement = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate,
                   'algorithm': algorithm,
                   'sampling_strategy': sampling_strategy,
                   'replacement': replacement}

    random_grid = {'algorithm': algorithm,
                   'replacement': replacement,
                   'learning_rate': learning_rate,
                   'n_estimators': n_estimators,
                   'sampling_strategy': sampling_strategy}
    
    def get_best_parameters(self,
                            features,
                            labels,
                            base_estimator=None,
                            n_iter=100,
                            cv=3,
                            verbose=2,
                            random_state=1,
                            n_jobs=-1):
        clf_random =\
            GridSearchCV(
                estimator=RUSBoostClassifier(),
                param_grid=self.random_grid,
                cv=cv,
                verbose=verbose,
                n_jobs=n_jobs
            )
        clf_random.fit(
            features,
            labels
        )

        return clf_random.best_params_

class RandomForestRandomizedCV:
    # from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 50)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    class_weight= ["balanced", "balanced_subsample"]
    criterion= ["gini", "entropy"]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': class_weight,
                   'criterion': criterion}
    
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

class NearestNeighborsRandomizedCV:
    n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 12, num = 3)]
    algorithm=['auto']
    metric=['minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    weights=['uniform', 'distance']

    random_grid = {'n_neighbors': n_neighbors,
                   'algorithm': algorithm,
                   'metric': metric}
    
    def get_best_parameters(self,
                            features,
                            labels,
                            n_iter=100,
                            cv=3,
                            verbose=2,
                            random_state=1,
                            n_jobs=-1):
        nn_random =\
            RandomizedSearchCV(
                estimator=KNeighborsClassifier(),
                param_distributions=self.random_grid,
                n_iter=n_iter,
                cv=cv,
                verbose=verbose,
                random_state=random_state,
                n_jobs=n_jobs
            )
        nn_random.fit(
            features,
            labels
        )

        return nn_random.best_params_