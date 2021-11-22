import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix

from project.ift6758.data.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration

#%%
random_state = np.random.RandomState(42)

#%%
data = pd.read_csv('project/ift6758/data/games_data/games_data_all_seasons.csv')

data = data[~(data['distance_net'].isna())]

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

#%%

x_train, x_valid, y_train, y_valid = train_test_split(train[['distance_net', 'angle_net']], train['is_goal'], random_state=random_state, stratify=train['is_goal'])

#%%
model = xgb.XGBClassifier(n_jobs=1)

model.fit(x_train, y_train)
pred = model.predict(x_valid)
print(confusion_matrix(y_valid, pred))
pred_proba = model.predict_proba(x_valid)

accuracy = np.sum(y_valid == pred)/len(pred)

#%%
labels = ['default_XGBoost']
probas = [pred_proba[:,1]]
#%%
plot_roc_curve(probas, y_valid, ['-'], labels)
plot_goal_rate(probas, y_valid, labels)
plot_cumulative_sum(probas, y_valid, labels)
plot_calibration(probas, y_valid, labels)
