#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from comet_ml import Experiment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from project.ift6758.data.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration, prep_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv("games_data/games_data_all_seasons.csv")
#data = pd.read_csv(os.path.join(os.environ.get('PATH_DATA'), 'games_data_all_seasons.csv'))
random_state = np.random.RandomState(42)
# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

#def prep_data(data_train):
    # Set seleceted features
selected_features = ['side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'team_side',
       'is_goal', 'distance_net', 'angle_net', 'time_since_pp_started',
       'time_since_pp_started', 'current_time_seconds', 'previous_event_x_coord', 'previous_event_y_coord',
       'current_friendly_on_ice', 'current_opposite_on_ice']

categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side']
train = prep_data(train_data, selected_features, categorical_features)

###
x_train, x_valid, y_train, y_valid = train_test_split(train[['distance_net', 'angle_net']], train['is_goal'], random_state=random_state, stratify=train['is_goal'])

###
clf = RandomForestClassifier(n_estimators=100)
#data = data_train[selected_features]
clf.fit(x_train, y_train)
predictions = clf.predict(x_valid)
print(confusion_matrix(y_valid, pred))
pred_proba = clf.predict_proba(x_valid)

accuracy = np.sum(y_valid == pred)/len(pred)
print(accuracy)
#print("Accuracy:",metrics.accuracy_score(y_test, pred_proba))
    # # Drop rows with NaN values
    # data = data.dropna(subset = selected_features)
    #
    # # Encoding categorical features into a one-hot encoding
    # categorical_features = ['side', 'shot_type', 'period', 'period_type', 'empty_net', 'previous_event_type']
    #
    # # Ecoding the features
    # for feature in categorical_features:
    #     one_hot_encoder = OneHotEncoder(sparse=False)
    #     encoding_df = data[[feature]]
    #
    #     one_hot_encoder.fit(encoding_df)
    #
    #     df_encoded = one_hot_encoder.transform(encoding_df)
    #
    #     df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)
    #
    #     # Drop original feature and add encoded features
    #     data.drop(columns=[feature], inplace=True)
    #
    #     data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
    #     # print("encoding:")
    #     # print(data.shape)
    #     # print(df_encoded.shape)
    #
    #
    #
    # # Split the data into features and labels for train and validation
    # x_train, x_valid, y_train, y_valid = train_test_split(data.drop(columns=['is_goal']), data['is_goal'], test_size=0.2, stratify=data['is_goal'])
    #
    # features_standardizing = [
    #     'coordinate_x', 'coordinate_y',
    #     'distance_net', 'angle_net', 'time_since_pp_started', 'current_time_seconds', 'current_friendly_on_ice', 'current_opposite_on_ice'
    # ]
    #
    # # normalization/standardization to features
    # scaler = StandardScaler()
    # # scaler = MinMaxScaler()
    # x_train[features_standardizing] = scaler.fit_transform(x_train[features_standardizing])
    # x_valid[features_standardizing] = scaler.fit_transform(x_valid[features_standardizing])
    #
    # # print(data.shape)
    # # print(x_train.shape)
    # # print(x_valid.shape)
    # # print(y_train.shape)
    # # print(y_valid.shape)
    #
    # return x_train, x_valid, y_train, y_valid, selected_features

