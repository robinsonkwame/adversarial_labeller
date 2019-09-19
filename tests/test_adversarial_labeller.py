from pathlib import Path
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from adversarial_labeller import AdversarialLabelerFactory


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

def keep_numeric_columns_only_replace_na(df: pd.DataFrame):
    return df.select_dtypes(include='number')\
             .fillna(value=-1, axis="columns")

MyBasicAdversarialPreprocessor = FunctionTransformer(
    keep_numeric_columns_only_replace_na,
    check_inverse=False,
    validate=False)

def test_adversarial_factory():
    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)

    pipeline =\
        AdversarialLabelerFactory(
            features = variables,
            labels = labels,
            inital_pipeline = MyBasicAdversarialPreprocessor
        ).fit_with_best_params()
