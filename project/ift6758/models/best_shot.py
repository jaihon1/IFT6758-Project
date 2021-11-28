# %%
import os
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from comet_ml import Experiment

# import keras
from sklearn.calibration import CalibrationDisplay
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from ift6758.utils.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration

# %%
# Load the data
# data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")
data = pd.read_csv("../data/games_data/games_data_all_seasons.csv")

# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300]  # remove outliers with value = inf

train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]


# %%
# features = [
#     'game_pk', 'side', 'event_index', 'event_type',
#     'shot_type', 'goal_strength', 'team_id', 'team_side', 'period',
#     'period_type', 'datetime', 'coordinate_x', 'coordinate_y',
#     'player_shooter', 'player_scorer', 'player_goalie', 'empty_net',
#     'is_goal', 'distance_net', 'angle_net', 'previous_event_type',
#     'previous_event_team', 'previous_event_x_coord',
#     'previous_event_y_coord', 'previous_event_period',
#     'previous_event_time_seconds', 'time_since_pp_started',
#     'current_time_seconds', 'current_friendly_on_ice',
#     'current_opposite_on_ice', 'shot_last_event_delta',
#     'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
# ]

def prep_data(data_train, bonus):
    if bonus:
        # Set selected features
        selected_features = [
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
        data = data_train[selected_features]

        # Drop rows with NaN values
        data = data.dropna(subset=selected_features)

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
            'is_goal', 'side',
            'shot_type', 'team_side', 'period',
            'period_type', 'coordinate_x', 'coordinate_y',
            'distance_net', 'angle_net', 'previous_event_type',
            'previous_event_x_coord',
            'previous_event_y_coord',
            'previous_event_time_seconds', 'shot_last_event_delta',
            'shot_last_event_distance', 'Change_in_shot_angle', 'Speed', 'Rebound'
        ]
        data = data_train[selected_features]

        # Drop rows with NaN values
        data = data.dropna(subset=selected_features)

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

    # Ecoding the features
    for feature in categorical_features:
        one_hot_encoder = OneHotEncoder(sparse=False)
        encoding_df = data[[feature]]

        one_hot_encoder.fit(encoding_df)

        df_encoded = one_hot_encoder.transform(encoding_df)

        df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)

        # Drop original feature and add encoded features
        data.drop(columns=[feature], inplace=True)

        data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    # Split the data into features and labels for train and validation
    x_train, x_valid, y_train, y_valid = train_test_split(data.drop(columns=['is_goal']), data['is_goal'],
                                                          test_size=0.2, stratify=data['is_goal'])

    # normalization/standardization to features
    scaler = StandardScaler()
    x_train[features_standardizing] = scaler.fit_transform(x_train[features_standardizing])
    x_valid[features_standardizing] = scaler.fit_transform(x_valid[features_standardizing])

    return x_train, x_valid, y_train, y_valid, selected_features


def find_optimal_threshold(predictions, true_y):
    scores = []

    for i in range(0, 100):
        # create a numpy array with the same shape as predictions
        masked_predictions = np.zeros(predictions.shape)
        for j, prediction in enumerate(predictions):
            if prediction <= i / 100:
                masked_predictions[j] = 0
            else:
                masked_predictions[j] = 1

        scores.append(f1_score(true_y, masked_predictions))

    print(np.max(scores), np.argmax(scores))

    threshold = np.argmax(scores) / 100
    masked_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        if prediction <= threshold:
            masked_predictions[i] = 0
        else:
            masked_predictions[i] = 1

    print(classification_report(true_y, masked_predictions))
    print(confusion_matrix(true_y, masked_predictions))


# %%
def train_model(x_train, x_valid, y_train, y_valid, class_weight, epoch, lr):
    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    filepath = 'nn.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    # Train the model
    model.fit(x_train, y_train, epochs=epoch, validation_data=(x_valid, y_valid), callbacks=callbacks,
              class_weight=class_weight, batch_size=128)

    # Evaluate the model
    model.evaluate(x_valid, y_valid)

    return model


# %%
def train_nn(x_train, x_valid, y_train, y_valid, features, comet=False):
    class_weight = {0: 1., 1: 3.}

    if comet:
        # Create experiment for comet
        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="ift6758-project",
            workspace="jaihon",
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True
        )
        experiment.log_parameters({'model': 'nn', 'feature': features, 'class_weight': class_weight})

    clf = train_model(x_train, x_valid, y_train, y_valid, class_weight, epoch=30, lr=0.001)

    if comet:
        model_name = 'nn' + '_'
        joblib.dump(clf, model_name + '.joblib')
        experiment.log_model(model_name, model_name + '.joblib')

    return clf


# %%
def main(data_train):
    TOGGLE_TRAIN = False

    print(os.environ.get("COMET_API_KEY"))

    x_train, x_valid, y_train, y_valid, features = prep_data(data_train, bonus=True)
    x_train_no_bonus, x_valid_no_bonus, y_train_no_bonus, y_valid_no_bonus, features_no_bonus = prep_data(data_train,
                                                                                                          bonus=False)

    if TOGGLE_TRAIN:
        clf = train_nn(x_train, x_valid, y_train, y_valid, features, comet=True)

    else:
        # File path

        # Load the model
        model = keras.models.load_model('../../models/nn/best_shot_nn_final.hdf5', compile=True)
        model1 = keras.models.load_model('../../models/nn/neuralnet_nobonus.hdf5', compile=True)
        model2 = keras.models.load_model('../../models/nn/neuralnet_no_dropout.hdf5', compile=True)

        # Generate predictions for samples
        predictions = model.predict(x_valid)
        predictions1 = model1.predict(x_valid_no_bonus)
        predictions2 = model2.predict(x_valid)

        find_optimal_threshold(predictions1, y_valid_no_bonus)

        plot_roc_curve([predictions, predictions1, predictions2], [y_valid, y_valid_no_bonus, y_valid], ['-', '-', '-'],
                       ['NeuralNet', 'NeuralNet_no_bonus', 'NeuralNet_no_dropout'])


if __name__ == "__main__":
    main(train_data)
# %%
