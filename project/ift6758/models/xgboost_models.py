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

from ift6758.utils.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration, prep_data, plot_relation

#%%
random_state = np.random.RandomState(42)

#%%
data = pd.read_csv('../data/games_data/games_data_all_seasons.csv')

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
print('Accuracy of default model:', accuracy)

#%%
# Experimenting with n_estimators
n_estimators = list(range(50, 450, 50))
kfold_split = StratifiedKFold(n_splits=3)
mean_acc = []
mean_prec = []
mean_f1 = []
for n_estimator in n_estimators:
    acc = []
    prec = []
    f1 = []
    for train_index, test_index in kfold_split.split(x_train, y_train):
        train_x, test_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
        model = xgb.XGBClassifier(n_estimators=n_estimator, n_jobs=1, random_state=random_state,
                                  use_label_encoder=False)
        model = model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        acc.append(accuracy_score(test_y, pred_y))
        prec.append(precision_score(test_y, pred_y))
        f1.append(f1_score(test_y, pred_y))
    mean_acc.append(sum(acc)/len(acc))
    mean_prec.append(sum(prec)/len(prec))
    mean_f1.append(sum(f1)/len(f1))

plot_relation(n_estimators, [mean_acc, mean_prec, mean_f1], ['Accuracy', 'Precision', 'F1 score'], 'n_estimator',
              'Performance metric')

#%%
# Experimenting with max_depth
max_depth = list(range(2, 20, 2))
kfold_split = StratifiedKFold(n_splits=3)
mean_acc = []
mean_prec = []
mean_f1 = []
for max_d in max_depth:
    acc = []
    prec = []
    f1 = []
    for train_index, test_index in kfold_split.split(x_train, y_train):
        train_x, test_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
        model = xgb.XGBClassifier(max_depth=max_d, n_jobs=1, random_state=random_state,
                                  use_label_encoder=False)
        model = model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        acc.append(accuracy_score(test_y, pred_y))
        prec.append(precision_score(test_y, pred_y))
        f1.append(f1_score(test_y, pred_y))
    mean_acc.append(sum(acc)/len(acc))
    mean_prec.append(sum(prec)/len(prec))
    mean_f1.append(sum(f1)/len(f1))

plot_relation(max_depth, [mean_acc, mean_prec, mean_f1], ['Accuracy', 'Precision', 'F1 score'], 'max_depth',
              'Performance metric')

#%%
# Experimenting with n_estimators
reg_lambda = np.arange(0, 2, 0.2)
kfold_split = StratifiedKFold(n_splits=3)
mean_acc = []
mean_prec = []
mean_f1 = []
for l in reg_lambda:
    acc = []
    prec = []
    f1 = []
    for train_index, test_index in kfold_split.split(x_train, y_train):
        train_x, test_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
        model = xgb.XGBClassifier(reg_lambda=l, n_jobs=1, random_state=random_state,
                                  use_label_encoder=False)
        model = model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        acc.append(accuracy_score(test_y, pred_y))
        prec.append(precision_score(test_y, pred_y))
        f1.append(f1_score(test_y, pred_y))
    mean_acc.append(sum(acc)/len(acc))
    mean_prec.append(sum(prec)/len(prec))
    mean_f1.append(sum(f1)/len(f1))

plot_relation(reg_lambda, [mean_acc, mean_prec, mean_f1], ['Accuracy', 'Precision', 'F1 score'], 'reg_lambda',
              'Performance metric')

#%%
# training using randomized search for section 5 question 2
kfold_split = StratifiedKFold(n_splits=3)
model_all_params = xgb.XGBClassifier(n_jobs=1, random_state=random_state, use_label_encoder=False)
hyperparams = {'n_estimators': [100, 150, 200, 250, 300], 'max_depth': [5, 10, 15], 'learning_rate': [3e-1, 3e-2, 3e-3],
               'reg_lambda': [0, 1, 1.2, 1.5]}
clf = RandomizedSearchCV(model_all_params, hyperparams, n_iter=20, scoring=['accuracy', 'precision', 'f1', 'roc_auc'],
                   random_state=np.random.RandomState(50), verbose=1, refit='roc_auc', cv=kfold_split)
clf.fit(x_train, y_train)
joblib.dump(clf, 'xgb_tuning2.joblib')
pred_proba_random = clf.predict_proba(x_valid)

#%%
# do some feature selection with Lasso
# we first plot the feature importance according to lasso
kfold_split = StratifiedKFold(n_splits=3)
x_train.columns = list(map(str, x_train.columns.values))
lasso = LassoCV(cv=kfold_split).fit(x_train, y_train)
feature_names = lasso.feature_names_in_
importance = np.abs(lasso.coef_)
plt.figure(figsize=(8, 6), dpi=200)
plt.bar(height=importance, x=feature_names)
plt.xticks(lasso.feature_names_in_, rotation='vertical', fontsize=9)
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

#%%
# select features according to lasso
sfm = SelectFromModel(lasso, threshold='median').fit(x_train, y_train)
features_selected = np.array(feature_names)[sfm.get_support()]

#%%
# train xbg with the feature selected by lasso
model_xgb = xgb.XGBClassifier(n_jobs=1, random_state=random_state, use_label_encoder=False)
hyperparams = {'n_estimators': [100, 150, 200, 250, 300], 'max_depth': [5, 10, 15],
               'learning_rate': [3e-1, 3e-2, 3e-3], 'reg_lambda': [0, 1, 1.2, 1.5]}

clf = RandomizedSearchCV(model_xgb, hyperparams, n_iter=10, scoring=['accuracy', 'precision', 'roc_auc', 'f1'],
                         random_state=np.random.RandomState(58), verbose=1, refit='roc_auc', cv=kfold_split)
clf.fit(x_train[features_selected], y_train)
joblib.dump(clf, 'xgb_lasso.joblib')
x_valid.columns = list(map(str, x_valid.columns.values))
pred_proba_lasso = clf.predict_proba(x_valid[features_selected])

#%%
# feature selection with mutual info
kfold_split = StratifiedKFold(n_splits=3)
model_xgb = xgb.XGBClassifier(n_jobs=1, random_state=random_state, use_label_encoder=False)
hyperparams = {'fs__k': [5, 7, 10, 15], 'xgb__n_estimators': [100, 150, 200], 'xgb__max_depth': [5, 10],
               'xgb__learning_rate': [3e-1, 3e-2, 3e-3], 'xgb__reg_lambda': [0, 1, 1.2, 1.5]}
pipe = Pipeline([('fs', SelectKBest(mutual_info_classif)),
                 ('xgb', model_xgb)])
clf = RandomizedSearchCV(pipe, hyperparams, n_iter=20, scoring=['accuracy', 'precision', 'f1'],
                         random_state=np.random.RandomState(58), verbose=1, refit='f1', cv=kfold_split)
clf.fit(x_train, y_train)
joblib.dump(clf, 'xgb_mutual.joblib')
pred_proba_mutual = clf.predict_proba(x_valid)

#%%
labels = ['default_XGBoost', 'tuning_XGBoost', 'lasso_XGBoost', 'mutual_XGBoost']
probas = [pred_proba[:, 1], pred_proba_random[:, 1], pred_proba_lasso[:, 1], pred_proba_mutual[:, 1]]
y_valids = [y_valid for i in range(4)]
#%%
plot_roc_curve(probas, y_valids, ['-', '-', '-', '-'], labels)
plot_goal_rate(probas, y_valids, labels)
plot_cumulative_sum(probas, y_valids, labels)
plot_calibration(probas, y_valids, labels)

