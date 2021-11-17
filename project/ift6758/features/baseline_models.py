import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


data = pd.read_csv('project/ift6758/data/games_data/games_data_all_seasons.csv')

data = data[~(data['distance_net'].isna())]

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
#%%
x_train, x_valid, y_train, y_valid = train_test_split(train['distance_net'], train['is_goal'], test_size=0.2, stratify=train['is_goal'])

x_train = x_train.array.reshape(-1, 1)
x_valid = x_valid.array.reshape(-1, 1)

#%%
clf = LogisticRegression()
clf.fit(x_train, y_train)

#%%
# Evaluate default logistic regression
pred = clf.predict(x_valid)
pred_proba = clf.predict_proba(x_valid)
accuracy_default = np.where(pred == y_valid, 1, 0).sum()/len(pred)
confusion_mat = confusion_matrix(y_valid, pred)

class1 = y_valid.sum()/len(y_valid)
class0 = (len(y_valid) - class1)/len(y_valid)

#%%
# ROC curve

random_model = DummyClassifier(strategy='stratified')
random_model.fit(x_train, y_train)
pred_random_model = random_model.predict_proba(x_valid)


def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')


plot_roc_curve(pred_random_model[:, 1], y_valid, '--', 'Random classifier')
plot_roc_curve(pred_proba[:, 1], y_valid, '-', 'Logistic Regression')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

#%%
# Compute percentile
percentile = np.arange(0, 100, 2)
percentile_pred = np.percentile(pred_proba[:, 1], percentile)

y_valid_df = pd.DataFrame(y_valid)

y_valid_df['bins_percentile'] = pd.cut(pred_proba[:, 1], percentile_pred)
goal_rate_by_percentile = y_valid_df.groupby(by=['bins_percentile']).apply(lambda g: g['is_goal'].sum()/len(g))

sns.set_theme()
g = sns.lineplot(x=percentile[:-1], y=goal_rate_by_percentile*100)
ax = g.axes
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
plt.xlim(100, 0)
plt.ylim(0, 100)
plt.xlabel('Shot probability model percentile')
plt.ylabel('Goals / (Shots + Goals)')
plt.show()

#%%
# plot cumulative proportion of goals as function of shot probability model percentile
total_number_goal = (y_valid_df['is_goal'] == 1).sum()
sum_goals_by_percentile = y_valid_df.groupby(by='bins_percentile').apply(lambda g: g['is_goal'].sum()/total_number_goal)
cum_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]

sns.set_theme()
g = sns.lineplot(x=percentile[1:], y=cum_sum_goals*100)
ax = g.axes
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
plt.xlim(100, 0)
plt.ylim(0, 100)
plt.xlabel('Shot probability model percentile')
plt.ylabel('Proportion')
plt.show()
