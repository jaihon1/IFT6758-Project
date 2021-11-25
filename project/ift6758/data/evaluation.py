#%%
import os
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from comet_ml import Experiment

# import keras
from keras.callbacks import ModelCheckpoint
from sklearn.calibration import CalibrationDisplay
from tensorflow import keras


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler


# %%
def prep_data(data_raw, bonus, model, std):
    if model == 'nn':
        if bonus:
            # Set seleceted features
            selected_features = [
                'game_pk',
                'is_goal', 'side',
                'shot_type', 'team_side', 'period',
                'period_type', 'coordinate_x', 'coordinate_y',
                'distance_net', 'angle_net', 'previous_event_type',
                'previous_event_x_coord',
                'previous_event_y_coord',
                'previous_event_time_seconds', 'time_since_pp_started',
                'current_time_seconds', 'current_friendly_on_ice',
                'current_opposite_on_ice', 'shot_last_event_delta',
                'shot_last_event_distance', 'Change_in_shot_angle', 'Speed', 'Rebound'
            ]
            data = data_raw[selected_features]

            # Drop rows with NaN values
            data = data.dropna(subset = selected_features)

            # Encoding categorical features into a one-hot encoding
            categorical_features = [
                'side',
                'shot_type', 'team_side', 'period',
                'period_type', 'previous_event_type',
                'current_friendly_on_ice', 'current_opposite_on_ice', 'Rebound'
            ]

            features_standardizing = [
                'coordinate_x', 'coordinate_y',
                'distance_net', 'angle_net',
                'previous_event_x_coord',
                'previous_event_y_coord',
                'previous_event_time_seconds', 'time_since_pp_started',
                'current_time_seconds',
                'shot_last_event_delta',
                'shot_last_event_distance', 'Change_in_shot_angle', 'Speed'
            ]

        else:
            # Set seleceted features
            selected_features = [
                'game_pk',
                'is_goal', 'side',
                'shot_type', 'team_side', 'period',
                'period_type', 'coordinate_x', 'coordinate_y',
                'distance_net', 'angle_net', 'previous_event_type',
                'previous_event_x_coord',
                'previous_event_y_coord',
                'previous_event_time_seconds', 'shot_last_event_delta',
                'shot_last_event_distance', 'Change_in_shot_angle', 'Speed', 'Rebound'
            ]
            data = data_raw[selected_features]

            # Drop rows with NaN values
            data = data.dropna(subset = selected_features)

            # Encoding categorical features into a one-hot encoding
            categorical_features = [
                'side',
                'shot_type', 'team_side', 'period',
                'period_type', 'previous_event_type',
                'Rebound'
            ]

            features_standardizing = [
                'coordinate_x', 'coordinate_y',
                'distance_net', 'angle_net',
                'previous_event_x_coord',
                'previous_event_y_coord',
                'previous_event_time_seconds',
                'shot_last_event_delta',
                'shot_last_event_distance', 'Change_in_shot_angle', 'Speed'
            ]


    # Ecoding features
    for feature in categorical_features:
        # print(f"Encoding categorical feature {feature}.")
        one_hot_encoder = OneHotEncoder(sparse=False)
        encoding_df = data[[feature]]

        one_hot_encoder.fit(encoding_df)

        df_encoded = one_hot_encoder.transform(encoding_df)

        df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)

        # Drop original feature and add encoded features
        data.drop(columns=[feature], inplace=True)

        data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
        # print("encoding:")
        # print(data.shape)
        # print(df_encoded.shape)


    # normalization/standardization to features
    if std:
        scaler = StandardScaler()
        data[features_standardizing] = scaler.fit_transform(data[features_standardizing])

    return data


# %%

def prediction_report(predictions, target, threshold):
    masked_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        if prediction <= threshold:
            masked_predictions[i] = 0
        else:
            masked_predictions[i] = 1

    print(classification_report(target, masked_predictions))
    print(confusion_matrix(target, masked_predictions))


# %%


def prepare(data, bonus, model_type, std):
    # Prepare data
    data = prep_data(data, bonus, model_type, std)

    # Split into train and test
    train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

    # Drop game pk
    train_data.drop(columns=['game_pk'], inplace=True)
    test_data.drop(columns=['game_pk'], inplace=True)

    # Split test data into input and target
    x_test, y_test = test_data.drop(columns=['is_goal']), test_data['is_goal']
    # x_train, y_train = train_data.drop(columns=['is_goal']), train_data['is_goal']

    return x_test, y_test

#%%
def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    sns.set_theme()
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')

# %%
'''
NEURAL NETWORK MODELS
'''

# Load the data
# data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")
data = pd.read_csv("games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

x_test, y_test = prepare(data, bonus=True, model_type='nn', std=False)
x_test_nobonus, y_test_nobonus = prepare(data, bonus=False, model_type='nn', std=False)


# Load the model
model = keras.models.load_model('nn_models_best/best_shot_nn_final.hdf5', compile = True)
model1 = keras.models.load_model('nn_models_best/unnecessary_truss_2939.hdf5', compile = True)
model2 = keras.models.load_model('nn_models_best/separate_alfalfa_7886.hdf5', compile = True)

# Generate predictions for samples
predictions = model.predict(x_test)
predictions1 = model1.predict(x_test_nobonus)
predictions2 = model2.predict(x_test)

prediction_report(predictions, y_test, threshold=0.33)
# prediction_report(predictions1, y_test_nobonus, threshold=0.33)
# prediction_report(predictions2, y_test, threshold=0.33)


# %%
'''
XGBOOST MODELS
'''

# Load the data
# data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")
data = pd.read_csv("games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

x_test, y_test = prepare(data, bonus=True, model_type='nn')
x_test_nobonus, y_test_nobonus = prepare(data, bonus=False, model_type='nn')

# Load the model
model = keras.models.load_model('nn_models_best/best_shot_nn_final.hdf5', compile = True)

# Generate predictions for samples
predictions = model.predict(x_test)

prediction_report(predictions, y_test, threshold=0.33)


# %%
