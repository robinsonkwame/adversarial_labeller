from pathlib import Path
import numpy as np
from functools import partial
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from adversarial_labeller import AdversarialLabelerFactory, Scorer


COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

def read_concat_and_label_test_train_data(test_filename="test.csv",
                                          train_filename="train.csv",
                                          data_dir="./tests/fixtures",
                                          y_columns=['Survived'],
                                          label_column="label",
                                          drop_columns=["PassengerId"]):
    drop_args = {
        "axis":"columns",
        "inplace": True
    }
    concat_args = {
        "axis": 0,
        "ignore_index": True,
        "sort": False
    }

    test_df = pd.read_csv(
        str(Path(data_dir)/Path(test_filename))
    )
    df =\
        pd.concat(
            (pd.read_csv(
                str(Path(data_dir)/Path(train_filename))), 
                test_df),
            **concat_args
        )
    df.drop(y_columns + drop_columns, **drop_args)
    df[label_column] = 0
    df.loc[len(test_df):, label_column] = 1
    return df

def get_variable_and_label_columns(df, label_column="label"):
    return df.drop(label_column, axis='columns'), df[label_column] 


def keep_numeric_columns_only_replace_na(df,
                                         all_columns=COLUMNS):
    transformed_df = None
    if isinstance(df, pd.DataFrame):
        transformed_df =\
            df.select_dtypes(include='number')\
              .fillna(value=-1, axis="columns")
    elif all_columns:
        transformed_df =\
            pd.DataFrame(df,
                         columns=all_columns)
        transformed_df =\
            transformed_df.select_dtypes(include='number')\
                          .fillna(value=-1, axis="columns")        

    return transformed_df

MyBasicAdversarialPreprocessor = FunctionTransformer(
    keep_numeric_columns_only_replace_na,
    check_inverse=False,
    validate=False)

def get_train_validate(train_filename="train.csv",
                       data_dir="./tests/fixtures",
                       fillna=True, 
                       fillna_value=0,
                       label_column="Survived",
                       drop_columns=["PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"],
                       train_ratio = 0.85):
    drop_args = {
        "axis":"columns",
        "inplace": True
    }

    fillna_args = {
        "inplace": True
    }

    train_df = pd.read_csv(
        str(Path(data_dir)/Path(train_filename))
    )

    train_df.drop(drop_columns, **drop_args)
    train_df.fillna(fillna_value, **fillna_args)

    # ... construct train, test and validate from repeated splits
    x_train, x_test, y_train, y_test =\
        train_test_split(train_df.drop(label_column, axis="columns"),
                         train_df.loc[:, label_column],
                         test_size = 1 - train_ratio)

    return {
        "train": {
            "labels": y_train,
            "data": x_train
        },
        "validate": {
            "labels": y_test,
            "data": x_test
        }
    }

def test_adversarial_factory():
    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)

    pipeline, flip_binary_predictions =\
        AdversarialLabelerFactory(
            features = variables,
            labels = labels,
            inital_pipeline = MyBasicAdversarialPreprocessor
        ).fit_with_best_params()

    scorer = Scorer(the_scorer=pipeline,
                    flip_binary_predictions=flip_binary_predictions)

    data = get_train_validate(fillna=True)
    clf = RandomForestClassifier(n_estimators=100)

    scores =\
        cross_val_score(
            X=data["train"]["data"],
            y=data["train"]["labels"],
            estimator=clf,
            scoring=scorer.grade,
            cv=5,
            n_jobs=1,
            verbose=2)
    average_score =\
        np.array(scores).mean()

    upper_bound =  0.71
    lower_bound = 0.67
    assert average_score < upper_bound and average_score > lower_bound,\
        "Expected 0.68 < average_score ({0:.2f}) < 0.69 but is not! Scores are {1}".format(
            average_score,
            scores
        )