from pathlib import Path
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

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

MyAdversarialPreprocessor = FunctionTransformer(
    keep_numeric_columns_only_replace_na,
    check_inverse=False,
    validate=False)

class MyAdversarialClassifier(LogisticRegression):
    pass

class AdversarialLabeller(MyAdversarialClassifier, TransformerMixin):
    def __init__(self):
        super().__init__()        

    def transform(self, X):
        return self.predict_proba(X)

if __name__ == "__main__":
    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)
    preprocessed_values = MyAdversarialPreprocessor.transform(variables)
    adversarial_labeller = AdversarialLabeller()
    adversarial_labeller.fit(preprocessed_values.values, labels.values)
    # e.g. ...
    #   adversarial_labeller.transform(preprocessed_values.values)