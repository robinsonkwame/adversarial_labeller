from pathlib import Path
import pandas as pd
from adversarial_labeller import (
    AdversarialRandomForestLabeller,
    AdversarialLogisticRegressionCVLabeller,
    AdversarialNearestNeighborLabeller,
    RandomForestRandomizedCV,
    NearestNeighborsRandomizedCV,
    RUSBoostRandomizedCV,
    AdversarialRUSBoostLabeller
    )
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier

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
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit   
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score

    df = read_concat_and_label_test_train_data()
    variables, labels = get_variable_and_label_columns(df)
    preprocessed_values =\
        MyBasicAdversarialPreprocessor.transform(
            variables
        )
    test_df, train_df = get_test_train_samples(
        preprocessed_values
    )

    # adversarial_labeller =\
    #     AdversarialLogisticRegressionCVLabeller(
    #         penalty='l2',
    #         class_weight='balanced',
    #         solver='liblinear',
    #         random_state=1,
    #         cv=5).fit(
    #     train_df.values,
    #     labels[train_df.index].values
    # )

    # print(
    #     "Accuracy: %0.2f (+/- %0.2f)" % (
    #         adversarial_labeller.scores_[1].mean(),
    #         adversarial_labeller.scores_[1].std() * 2)
    #     )
    # print(
    #     "Validation Accuracy: %0.2f" % (
    #         accuracy_score(
    #             labels[test_df.index],
    #             adversarial_labeller.predict(
    #                 test_df.values
    #             )
    #         )
    #     )
    # )
    # print(
    #     classification_report(
    #         y_true= labels[test_df.index],
    #         y_pred= adversarial_labeller.predict(
    #                     test_df.values
    #                 )
    #     )
    # )

    # adversarial_labeller = AdversarialRandomForestLabeller(
    #     n_estimators=50
    # )

    # adversarial_labeller =\
    #  AdversarialNearestNeighborLabeller(
    #     n_neighbors=2,
    #     algorithm='ball_tree'
    # )

    # cross_validator =\
    #     StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    # scores =\
    #     cross_val_score(adversarial_labeller,
    #                     train_df.values,
    #                     labels[train_df.index].values,
    #                     cv=cross_validator)

    # fit_params =\
    #     RandomForestRandomizedCV().get_best_parameters(
    #         features=train_df.values,
    #         labels=labels[train_df.index]
    #     )

    # adversarial_labeller = AdversarialRandomForestLabeller(
    #     fit_params=fit_params
    # )

    # fit_params =\
    #     NearestNeighborsRandomizedCV().get_best_parameters(
    #         features=train_df.values,
    #         labels=labels[train_df.index],
    #     )

    # adversarial_labeller = AdversarialNearestNeighborLabeller(
    #     fit_params=fit_params
    # ).fit(
    #     train_df.values,
    #     labels[train_df.index].values
    # )


    # adversarial_labeller = BalancedRandomForestClassifier(
    #     n_estimators=100,
    #     random_state=0
    # ).fit(
    #     train_df.values,
    #     labels[train_df.index].values
    # )


    fit_params =\
        RUSBoostRandomizedCV().get_best_parameters(
            n_iter=50,
            features=train_df.values,
            labels=labels[train_df.index],
        )

    adversarial_labeller = AdversarialRUSBoostLabeller(
        fit_params=fit_params
    ).fit(
        train_df.values,
        labels[train_df.index].values
    )

    # print(
    #     "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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

    # adversarial_labeller.fit(
    #     train_df.values,
    #     labels[train_df.index].values
    # )

    # rf = RandomForestClassifier()
    # rf.fit(train_df.values,
    #        labels[train_df.index])

    # # ----
    
    # # # vectorizer 
    # # vectorizer = CountVectorizer()
    # # extra_features = vectorizer.fit_transform(
    # #     df.Name.str.replace(",", "_is_a_last_name")
    # # )

    # preprocessed_values = MyBasicAdversarialPreprocessor.transform(variables)

    # y_score = adversarial_labeller.transform(test_df.values)
    # roc_auc_score(
    #     labels[test_df.index].values,
    #     y_score[:,1]
    # )
    # # Out[136]: 0.5286266924564796

    # # try with vtreat as the preprocessor
    # import vtreat

    # plan = vtreat.BinomialOutcomeTreatment(outcome_name="label",
    #                                        outcome_target=True)

    # train_df = variables.sample(**sample_args)
    # test_df = variables.drop(train_df.index)

    # cross_frame = plan.fit_transform(
    #     train_df,
    #     labels[train_df.index].values
    # )
    # adversarial_labeller = AdversarialRandomForestLabeller(
    #     args_for_randomforest={
    #         "n_estimators":100
    #     }
    # )        

    # adversarial_labeller = AdversarialExtraTreesLabeller(
    #     args_for_classifer={
    #         "n_estimators":100
    #     }
    # )

    # adversarial_labeller.fit(
    #     cross_frame.values,
    #     labels[train_df.index].values
    # )

    # y_score = adversarial_labeller.transform(
    #     plan.transform(test_df)
    # )
    # roc_auc_score(
    #     labels[test_df.index].values,
    #     y_score[:, 1]
    # )
    # #Out[123]: 0.5499032882011605

    # #  Need to create adversarial classifier by
    # # classifier(Classifier)
    # # def score(X, y, sample_weight):
    # #   and we can assign a sample weight via probability 
    # # instance is a test instnace.

    # # this might not be well founded, but can be checked
    # # against Kaggle submission (e.g., score should be similar)

    # # Okay, so I think I can 
    # #   use fit_params in a CV like thing
    # # and also provide sample_weight as well as score_params

    # df =\
    #     pd.read_csv('./test/fixtures/train.csv')\
    #       .drop("PassengerId", axis="columns")
    # preprocessed_df = MyBasicAdversarialPreprocessor.transform(df)

    # variables = preprocessed_df.drop("Survived", axis="columns")
    # labels = preprocessed_df["Survived"]

    # rf_train_df = variables.sample(**sample_args)
    # rf_test_df = variables.drop(rf_train_df.index)

    # new_cross_frame =\
    #     plan.transform(
    #         df.loc[rf_train_df.index]
    #     )

    # rf = RandomForestClassifier()
    # rf.fit(rf_train_df.values,
    #        labels[rf_train_df.index],
    #        sample_weight=\
    #            adversarial_labeller.transform(
    #             new_cross_frame)[:, 1]
    # )

    # rf2 = RandomForestClassifier()
    # rf2.fit(rf_train_df.values,
    #         labels[rf_train_df.index]
    # )

    # test_cross_frame =\
    #     plan.transform(
    #         df.loc[rf_test_df.index]
    #     )
    # sample_weights =\
    #     adversarial_labeller.transform(
    #         test_cross_frame)[:, 1]

    # #  Test on hold out set
    # from sklearn.metrics import accuracy_score
    # print(
    #     accuracy_score(labels[rf_test_df.index],
    #                    rf.predict(rf_test_df.values),
    #                    sample_weight=\
    #                     adversarial_labeller.transform(
    #                         test_cross_frame)[:, 1]
    #     )
    # )

    # print(
    #     accuracy_score(survived[test_df.index],
    #                    rf.predict(
    #                        plan.transform(test_df))
    #     )
    # )

    # from sklearn.metrics import accuracy_score
    # print(
    #     accuracy_score(labels[rf_test_df.index].values,
    #                    rf2.predict(rf_test_df.values),
    #                    sample_weight=sample_weights
                        
    #     )
    # )
    # # Gets .763 (higher)

    # print(
    #     accuracy_score(labels[rf_test_df.index].values,
    #                    rf2.predict(rf_test_df.values)
    #     )
    # )
    # # Gets .76 on hold out
    # # Get 0.64593 on test

    # mask  = sample_weights == 1
    # print(
    #     accuracy_score(labels[rf_test_df.index][mask].values,
    #                    rf2.predict(rf_test_df[mask].values)
    #     )
    # )
    # # Gets 0.6875 on score check (0.04 off from test)


    # titanic_df =\
    #     pd.read_csv('./test/fixtures/test.csv')\
    #       .drop("PassengerId", axis="columns")
    # titanic_df.index += 892
    # preprocessed_titanic_df = MyBasicAdversarialPreprocessor.transform(titanic_df)


    # predictions =\
    #     rf.predict(preprocessed_titanic_df)
    
    # outputs =\
    #     pd.DataFrame(
    #         {'PassengerId': preprocessed_titanic_df.index,
    #          'Survived': predictions}
    #     )
    # outputs.to_csv('titanic_output.csv', index=False)
    # # gets 0.333 on kaggle, against label for test instance or not
    
    # # rewrote to be against train.csv, with vtreat sample weights
    # # under fit
    # # gets 0.59808

    # predictions2 =\
    #     rf2.predict(preprocessed_titanic_df)
    
    # outputs2 =\
    #     pd.DataFrame(
    #         {'PassengerId': preprocessed_titanic_df.index,
    #          'Survived': predictions2}
    #     )
    # outputs2.to_csv('titanic_output2.csv', index=False)
    # #  gets 0.40191 on kaggle