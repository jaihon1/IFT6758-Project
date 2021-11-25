import os
import joblib

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from project.ift6758.data.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration, prep_data, plot_relation

#%%
random_state = np.random.RandomState(42)

#%%
data = pd.read_csv(os.path.join(os.environ.get('PATH_DATA'), 'games_data_all_seasons.csv'))

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

# impute infinity value in speed column
max_speed = data['Speed'].replace([np.inf], -np.inf).max()
data['Speed'] = data['Speed'].replace([np.inf], max_speed)

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
#%%
selected_features = ['side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'is_goal',
                     'team_side', 'distance_net', 'angle_net', 'previous_event_type', 'time_since_pp_started',
                     'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                     'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                     'shot_last_event_delta', 'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
                    ]

categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side', 'previous_event_type']
scale_features = ['coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'time_since_pp_started',
                  'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                  'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                  'shot_last_event_delta', 'shot_last_event_distance', 'Change_in_shot_angle', 'Speed']

train = prep_data(train, selected_features, categorical_features, norm=scale_features)
#%%

x_train, x_valid, y_train, y_valid = train_test_split(train.drop(columns=['is_goal']), train['is_goal'], random_state=random_state, stratify=train['is_goal'])

#%%
model = xgb.XGBClassifier(n_jobs=1, random_state=random_state, use_label_encoder=False)

model = model.fit(x_train[['distance_net', 'angle_net']], y_train)
# %%
pred = model.predict(x_valid[['distance_net', 'angle_net']])
joblib.dump(model, 'default_xgb.joblib')
print('Confusion matrix of default model:', confusion_matrix(y_valid, pred))
pred_proba = model.predict_proba(x_valid[['distance_net', 'angle_net']])

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

