# adversarial_labeller
---

Adversarial labeller is a sklearn compatible labeller that scores instances as belonging to the test dataset or not to help model selection under data drift. Adversarial labeller is distributed under the MIT license.

## Installation

*Dependencies*

Adversarial validator requires:
* Python (>= 3.7)
* scikit-learn (>= 0.21.0)
* [imbalanced learn](>= 0.5.0)
* [pandas](>= 0.25.0)

*User installation*

The easiest way to install adversarial validator is using 
```pip
pip install adversarial_labeller
```

*Example Usage*
```python
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from adversarial_labeller import AdversarialLabelerFactory, Scorer

# Our blob data generation parameters for this example
number_of_samples = 1000
number_of_test_samples = 300

# Generate 1d blob data and label a portion as test data
# ... 1d blob data can be visualized as a rug plot
variables, labels = \
  make_blobs(
    n_samples=number_of_samples,
    centers=2,
    n_features=1,
    random_state=0
)

df = pd.DataFrame(
  {
    'independent_variable':variables.flatten(),
    'dependent_variable': labels,
    'label': 0  #  default to train data
  }
)
test_indices = df.index[-number_of_test_samples:]
train_indices = df.index[:-number_of_test_samples]

df.loc[test_indices,'label'] = 1  # ... now we mark instances that are test data

# Now perturb the test samples to simulate data drift/different test distribution
df.loc[test_indices, "independent_variable"] +=\
  np.std(df.independent_variable)

# ... now we have an example of data drift where adversarial labeling can be used to better estimate the actual test accuracy

features_for_labeller = df.independent_variable
labels_for_labeller = df.label

pipeline, flip_binary_predictions =\
    AdversarialLabelerFactory(
        features = features_for_labeller,
        labels = labels_for_labeller,
        run_pipeline = False
    ).fit_with_best_params()

scorer = Scorer(the_scorer=pipeline,
                flip_binary_predictions=flip_binary_predictions)

# Now we evaluate a classifer on training data only, but using
# our fancy adversarial labeller
_X = df.loc[train_indices]\
       .independent_variable\
       .values\
       .reshape(-1,1)

_X_test = df.loc[test_indices]\
            .independent_variable\
            .values\
            .reshape(-1,1)

# ... sklearn wants firmly defined shapes
clf_adver = RandomForestClassifier(n_estimators=100, random_state=1)
adversarial_scores =\
    cross_val_score(
        X=_X,
        y=df.loc[train_indices].dependent_variable,
        estimator=clf_adver,
        scoring=scorer.grade,
        cv=10,
        n_jobs=-1,
        verbose=1)
# ... and we get ~ 0.70
average_adversarial_score =\
    np.array(adversarial_scores).mean()

# ... let's see how this compares with normal cross validation
clf = RandomForestClassifier(n_estimators=100, random_state=1)
scores =\
    cross_val_score(
        X=_X,
        y=df.loc[train_indices].dependent_variable,
        estimator=clf,
        cv=10,
        n_jobs=-1,
        verbose=1)

# ... and we get ~ 0.92
average_score =\
    np.array(scores).mean()

# now let's see how this compares with the actual test score
clf_all = RandomForestClassifier(n_estimators=100, random_state=1)
clf_all.fit(_X,
            df.loc[train_indices].dependent_variable)

# ... actual test score is 0.70
actual_score =\
  accuracy_score(
    clf_all.predict(_X_test),
    df.loc[test_indices].dependent_variable
  )

adversarial_result = abs(average_adversarial_score - actual_score)
print(f"... adversarial labelled cross validation was {adversarial_result:.2f} points different than actual.")  # ... 0.00 points

cross_val_result = abs(average_score - actual_score)
print(f"... regular validation was {cross_val_result:.2f} points different than actual.")  # ... 0.23 points
```
