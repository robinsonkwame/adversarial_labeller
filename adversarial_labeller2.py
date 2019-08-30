from pathlib import Path
import pandas as pd
from adversarial_labeller import (
    RUSBoostRandomizedCV,
    AdversarialRUSBoostLabeller
    )
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

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
    test_df, train_df = get_test_train_samples(
        preprocessed_values
    )

    fit_params =\
        RUSBoostRandomizedCV().get_best_parameters(
            n_iter=5,
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