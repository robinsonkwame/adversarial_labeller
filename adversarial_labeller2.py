from pathlib import Path
import pandas as pd
from adversarial_labeller import AdversarialRandomForestLabeller
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer


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

if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    sample_args = {
        "frac": 0.85,
        "random_state": 1
    }

    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)

    # vectorizer 
    vectorizer = CountVectorizer()
    extra_features = vectorizer.fit_transform(
        df.Name\
          .str.replace(' ','#')\
          .str.replace(',',',#').str.split('#')
    )
    # ... todo: figure out way to simply take Mr/Mrs fact of
    # as well as proxy for count of last name on ticket
    #       so this is a fast way but some variables need to be
    # combined so that vtreat can handle as the same thing, est importance?

    preprocessed_values = MyBasicAdversarialPreprocessor.transform(variables)

    train_df = preprocessed_values.sample(**sample_args)
    test_df = preprocessed_values.drop(train_df.index)

    adversarial_labeller = AdversarialRandomForestLabeller()
    adversarial_labeller.fit(
        train_df.values,
        labels[train_df.index].values
    )
    y_score = adversarial_labeller.transform(test_df.values)
    roc_auc_score(
        labels[test_df.index].values,
        y_score[:,1]
    )
    # Out[136]: 0.5286266924564796

    # try with vtreat as the preprocessor
    import vtreat

    plan = vtreat.BinomialOutcomeTreatment(outcome_name="label",
                                           outcome_target=True)

    train_df = variables.sample(**sample_args)
    test_df = variables.drop(train_df.index)

    cross_frame = plan.fit_transform(
        train_df,
        labels[train_df.index].values
    )
    adversarial_labeller = AdversarialRandomForestLabeller()
    adversarial_labeller.fit(
        cross_frame.values,
        labels[train_df.index].values
    )

    y_score = adversarial_labeller.transform(
        plan.transform(test_df)
    )
    roc_auc_score(
        labels[test_df.index].values,
        y_score[:, 1]
    )
    #Out[123]: 0.5499032882011605