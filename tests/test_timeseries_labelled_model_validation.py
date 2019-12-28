from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from adversarial_labeller import AdversarialLabelerFactory, Scorer
import numpy as np
import pandas as pd

scoring_metric = r2_score

config =\
    {
        "file_name": "aaap.us.txt",
        "sep":",",
        "root_directory": "tests",
        "file_path": "./fixtures",
        "type": "csv"
    }

def test_timeseries_labelled_model_validation():
    df = None
    if 'csv' == config['type']:
        df = pd.read_csv(
            '/'.join(
                [config['root_directory'],
                 config['file_path'],
                 config['file_name']]
            ),
            sep=config['sep']
        )
    df['Datetime'] =\
        (df.Date + " " + df.Time).astype(
            'datetime64[ns]'
        )
    df.set_index('Datetime', inplace= True)
    df.drop(['Date', 'Time'],
            inplace= True,
            axis='columns')
    y_column = ['Close']

    number_of_test_instances = 100
    X_train = df[:-number_of_test_instances].drop(y_column, axis='columns')
    y_train = df[:-number_of_test_instances][y_column].values.ravel()
    X_test = df[-number_of_test_instances:].drop(y_column, axis='columns')
    y_test = df[-number_of_test_instances:][y_column].values.ravel()

    cv = TimeSeriesSplit(n_splits=10).split(X_train)
    clf = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)

    score =\
        cross_val_score(
            estimator=clf,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=make_scorer(scoring_metric)).mean()

    print(f"\t cross validation score: {score}")

    variables = df.drop(y_column, axis='columns')
    labels = pd.DataFrame(
        {"labels": np.zeros(len(df)).astype('int')}
    )
    labels[-number_of_test_instances:] = 1
    labels.index = variables.index

    pipeline, flip_binary_predictions =\
        AdversarialLabelerFactory(
            features = variables,
            labels = labels,
            run_pipeline=False
        ).fit_with_best_params()

    clf = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)        

    scorer = Scorer(the_scorer=pipeline,
                    flip_binary_predictions=flip_binary_predictions,
                    metric=r2_score,
                    predict_proba_threshold=0.40)
    1/0

    cv = TimeSeriesSplit(n_splits=10).split(X_train)
    adversarial_score =\
        cross_val_score(
            estimator=clf,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scorer.grade).mean()

    print(f"\t adversarial cross validation score: {adversarial_score}")

    clf = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    holdout_score =\
        scoring_metric(
            y_test,
            clf.predict(X_test)
        )
    print(f"\t actual hold out score: {holdout_score}")

    assert abs(holdout_score-adversarial_score) < abs(holdout_score-score),\
        (f"Whoops! Holdout score ({holdout_score:.3f}) is closer to "
         f"non-adversarial score ({score:.3f} vs {adversarial_score:.3f})!")
