from pathlib import Path
import pandas as pd
from adversarial_labeller import (
    RUSBoostRandomizedCV,
    AdversarialRUSBoostLabeller
    )
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_target_labels(test_filename="test.csv",
                      train_filename="train.csv",
                      data_dir="./test/fixtures",
                      y_columns=['Survived']):
    df =\
        pd.read_csv(
            str(Path(data_dir)/Path(train_filename))
        )[y_columns]
    return df

def get_train_validate(train_filename="train.csv",
                            data_dir="./test/fixtures",
                            fillna=True,
                            fillna_value=0,
                            label_column="Survived",
                            drop_columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"],
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

def read_concat_and_label_test_train_data(test_filename="test.csv",
                                          train_filename="train.csv",
                                          data_dir="./test/fixtures",
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

def keep_numeric_columns_only_replace_na(df: pd.DataFrame):
    return df.select_dtypes(include='number')\
             .fillna(value=-1, axis="columns")

MyBasicAdversarialPreprocessor = FunctionTransformer(
    keep_numeric_columns_only_replace_na,
    check_inverse=False,
    validate=False)

def get_test_train_samples(df,
                           sample_args=None):
    if not sample_args:
        sample_args =\
            {
                "frac": 0.85,
                "random_state": 1
            }
    train_df = df.sample(**sample_args)
    test_df = df.drop(train_df.index)

    return train_df, test_df  # should have an API manager?

if __name__ == "__main__":
    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)
    preprocessed_values =\
        MyBasicAdversarialPreprocessor.transform(
            variables
        )
    train_df, test_df = get_test_train_samples(
        preprocessed_values
    )

    fit_params =\
        RUSBoostRandomizedCV().get_best_parameters(
            n_iter=1,
            features=train_df.values,
            labels=labels[train_df.index],
        )

    adversarial_labeller = AdversarialRUSBoostLabeller(
        fit_params=fit_params
    ).fit(
        train_df.values,
        labels[train_df.index].values
    )

    adversarial_labeller.maximize_binary_validation_accuracy(
        test_df.values,
        labels[test_df.index]
    )

    print(
        "Validation Accuracy: %0.2f" % (
            accuracy_score(
                labels[test_df.index],
                adversarial_labeller.predict(
                    test_df.values
                )
            )
        )
    )

    print(
        classification_report(
            y_true= labels[test_df.index],
            y_pred= adversarial_labeller.predict(
                        test_df.values
                    )
        )
    )

    # # Now we train a stock random forest classifiers and test
    # # its accuracy against the test labeled hold out and compare to Kaggle score

    # # ... first need to read in Survived values for train_df and use in fit
    data = get_train_test_validate(fillna=True)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data["train"]["data"],
            data["train"]["labels"])
    
    # ... now we mark those instances as in test or not
    labelled_test_mask =\
        adversarial_labeller.predict(
            titanic_train_df.values
        ) == 1

    # # and use that test labeled set as a validation set, compare to Kaggle
    # accuracy_score(
    #     y_true= labels[test_df.index & labelled_test_mask],
    #     y_pred= clf.predict(test_df[labelled_test_mask].values)
    # )
