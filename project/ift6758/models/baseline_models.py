# %%
import os
import joblib

import pandas as pd
import numpy as np
from comet_ml import Experiment
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ift6758.utils.utils import *

# %%

data = pd.read_csv(os.path.join(os.environ.get('PATH_DATA'), 'games_data_all_seasons.csv'))

data = data[~(data['distance_net'].isna())]

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
# %%
x_train, x_valid, y_train, y_valid = train_test_split(train.drop(columns=['is_goal']), train['is_goal'], test_size=0.2,
                                                      stratify=train['is_goal'])


def train_logistic(X, y, features, comet=False):
    """
    Train a logistic regression and log the experiment on comet if asked
    Args:
        X: pd.DataFrame; the training examples
        y: pd.Series; the training labels
        features: list of strings; name of the features to use for training
        comet: bool; True if we want to log experiment on comet

    Returns: the trained classifier

    """
    if comet:
        # Create experiment for comet
        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="ift6758-project",
            workspace="jaihon",
        )
        experiment.log_parameters({'model': 'regression', 'feature': features})
    X = X[features]
    X = X[features].abs()

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    if comet:
        model_name = 'regression_' + '_'.join(features)
        joblib.dump(clf, model_name + '.joblib')
        experiment.log_model(model_name, model_name + '.joblib')

    return clf


# %%
# train the logisitc regressions on different features
comet = False
clf_distance = train_logistic(x_train, y_train, ['distance_net'], comet)
clf_angle = train_logistic(x_train, y_train, ['angle_net'], comet)
clf_both = train_logistic(x_train, y_train, ['distance_net', 'angle_net'], comet)

# %%
# prediction of all models on validation set
pred_proba_distance = clf_distance.predict_proba(x_valid[['distance_net']])
pred_proba_angle = clf_angle.predict_proba(x_valid[['angle_net']].abs())
pred_proba_both = clf_both.predict_proba(x_valid[['distance_net', 'angle_net']].abs())

# %%
# Evaluate default logistic regression
pred = clf_distance.predict(x_valid[['distance_net']])
accuracy_default = np.where(pred == y_valid, 1, 0).sum() / len(pred)
confusion_mat = confusion_matrix(y_valid, pred)

class1 = y_valid.sum() / len(y_valid)
class0 = (len(y_valid) - class1) / len(y_valid)

# %%
# set up the plots
np.random.seed(42)
pred_random_model = np.random.uniform(size=len(y_valid))

pred_proba = [pred_random_model, pred_proba_distance[:, 1], pred_proba_angle[:, 1], pred_proba_both[:, 1]]
labels = ['Random classifier', 'Regression distance', 'Regression angle', 'Regression both angle and distance']
linestyles = ['--', '-', '-', '-']
y_valids = [y_valid for i in range(3)]
#%%
# plot the 4 curves asked in part 3 question 2
plot_roc_curve(pred_proba, y_valids, linestyles, labels)
plot_goal_rate(pred_proba, y_valids, labels)
plot_cumulative_sum(pred_proba, y_valids, labels)
plot_calibration(pred_proba, y_valids, labels)
