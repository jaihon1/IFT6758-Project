#%%
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
from comet_ml import Experiment


#%%

data = pd.read_csv('../../../project/ift6758/data/games_data/games_data_all_seasons.csv')

data = data[~(data['distance_net'].isna())]

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
#%%
x_train, x_valid, y_train, y_valid = train_test_split(train.drop(columns=['is_goal']), train['is_goal'], test_size=0.2, stratify=train['is_goal'])

def train_logistic(X, y, features):
    X = X[features]

    if features == ['angle_net']:
        X = X[features].abs()
    clf = LogisticRegression()
    clf.fit(X, y)

    return clf

x_train = x_train.array.reshape(-1, 1)
x_valid = x_valid.array.reshape(-1, 1)

#%%
clf_distance = train_logistic(x_train, y_train, ['distance_net'])
clf_angle = train_logistic(x_train, y_train, ['angle_net'])
clf_both = train_logistic(x_train, y_train, ['distance_net', 'angle_net'])

#%%
pred_proba_distance = clf_distance.predict_proba(x_valid[['distance_net']])
pred_proba_angle = clf_angle.predict_proba(x_valid[['angle_net']])
pred_proba_both = clf_both.predict_proba(x_valid[['distance_net', 'angle_net']])

#%%
# Evaluate default logistic regression
pred = clf_distance.predict(x_valid[['distance_net']])
accuracy_default = np.where(pred == y_valid, 1, 0).sum()/len(pred)
confusion_mat = confusion_matrix(y_valid, pred)

class1 = y_valid.sum()/len(y_valid)
class0 = (len(y_valid) - class1)/len(y_valid)

#%%
# ROC curve

#random_model = DummyClassifier(strategy='uniform')
#random_model.fit(x_train, y_train)
np.random.seed(42)
pred_random_model = np.random.uniform(size=len(y_valid))


def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')


plot_roc_curve(pred_random_model, y_valid, '--', 'Random classifier')
plot_roc_curve(pred_proba_distance[:, 1], y_valid, '-', 'Regression distance')
plot_roc_curve(pred_proba_angle[:, 1], y_valid, '-', 'Regression angle')
plot_roc_curve(pred_proba_both[:, 1], y_valid, '-', 'Regression both angle and distance')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

#%%
# Compute percentile
def plot_goal_rate(proba, label):
    percentile = np.arange(0, 100, 2)
    percentile_pred = np.percentile(proba, percentile)

    y_valid_df = pd.DataFrame(y_valid)

    y_valid_df['bins_percentile'] = pd.cut(proba, percentile_pred, duplicates='drop')
    goal_rate_by_percentile = y_valid_df.groupby(by=['bins_percentile']).apply(lambda g: g['is_goal'].sum()/len(g))

    if len(percentile_pred)-len(goal_rate_by_percentile) == 1:
        percentile = percentile[:-1]
    else:
        percentile = percentile[:-2]
    g = sns.lineplot(x=percentile, y=goal_rate_by_percentile*100, label=label)
    ax = g.axes
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    return y_valid_df

valid_random = plot_goal_rate(pred_random_model, 'Random classifier')
valid_distance = plot_goal_rate(pred_proba_distance[:,1], 'Regression distance')
valid_angle = plot_goal_rate(pred_proba_angle[:,1], 'Regression angle')
valid_both = plot_goal_rate(pred_proba_both[:,1], 'Regression both distance and angle')
plt.xlim(100, 0)
plt.ylim(0, 100)
plt.xlabel('Shot probability model percentile')
plt.ylabel('Goals / (Shots + Goals)')
plt.show()

#%%
# plot cumulative proportion of goals as function of shot probability model percentile
def plot_cumulative_sum(y_valid_df, label):
    percentile = np.arange(0,100,2)
    total_number_goal = (y_valid == 1).sum()
    sum_goals_by_percentile = y_valid_df.groupby(by='bins_percentile').apply(lambda g: g['is_goal'].sum()/total_number_goal)
    cum_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]

    sns.set_theme()
    if len(percentile)-len(cum_sum_goals) == 1:
        percentile = percentile[:-1]
    else:
        percentile = percentile[:-2]
    g = sns.lineplot(x=percentile, y=cum_sum_goals*100, label=label)
    ax = g.axes
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
plot_cumulative_sum(valid_random, 'Random classifier')
plot_cumulative_sum(valid_distance, 'Regression distance')
plot_cumulative_sum(valid_angle, 'Regression angle')
plot_cumulative_sum(valid_both, 'Regression both distance and angle')
plt.xlim(100, 0)
plt.ylim(0, 100)
plt.xlabel('Shot probability model percentile')
plt.ylabel('Proportion')
plt.show()


