from pathlib import Path
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression

def read_concat_and_label_test_train_data(test_filename="test.csv",
                                          train_filename="train.csv",
                                          data_dir="./test/fixtures",
                                          y_columns=['Survived'],
                                          label_column="label"):
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
    df.drop(y_columns, **drop_args)
    df[label_column] = 0
    df.loc[len(test_df):, label_column] = 1
    return df

def get_variable_and_label_columns(df, label_column="label"):
    return df.drop(label_column, axis='column'), df[label_column] 

class AdversarialClassifier(LogisticRegression):
    # note:  if we wanted this to be dynamic we'd use dependency injection
    pass

class AdversarialLabeller(TransformerMixin, AdversarialClassifier):
    def __init__(self):
        pass

    def transform(self, X):
        return self.predict_proba(X)

#todo: make sure this works as expected:
# 
# can instantiate an AdversarialLabeller with, say,
# a randomforest, use fit with appropriately transformed data
# then can use transform and get probabilities, would be pretty cool
# if this simple