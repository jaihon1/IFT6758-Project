import os

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix

from project.ift6758.data.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration, prep_data

#%%
random_state = np.random.RandomState(42)

#%%
data = pd.read_csv(os.path.join(os.environ.get('PATH_DATA'), 'games_data_all_seasons.csv'))

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
#%%
selected_features = ['side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'is_goal',
                     'team_side', 'distance_net', 'angle_net', 'previous_event_type', 'time_since_pp_started',
                     'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                     'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                     'shot_last_event_delta', 'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
                    ]

categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side', 'previous_event_type']

train = prep_data(train, selected_features, categorical_features)

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

#%%
plot_roc_curve(probas, y_valid, ['-'], labels)
plot_goal_rate(probas, y_valid, labels)
plot_cumulative_sum(probas, y_valid, labels)
plot_calibration(probas, y_valid, labels)

