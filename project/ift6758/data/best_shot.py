#%%
import os
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from comet_ml import Experiment

# import keras
from keras.callbacks import ModelCheckpoint
from tensorflow import keras


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%
# Load the data
# data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")
data = pd.read_csv("games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300]

train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]


#%%
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

def prep_data(data_train):
    # Set seleceted features
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

    # selected_features = ['is_goal', 'side', 'shot_type',
    #    'period', 'period_type', 'coordinate_x', 'coordinate_y', 'empty_net',
    #     'distance_net', 'angle_net', 'previous_event_type',
    #    'time_since_pp_started', 'current_time_seconds',
    #    'current_friendly_on_ice', 'current_opposite_on_ice'
    # ]

    data = data_train[selected_features]

    # Drop rows with NaN values
    data = data.dropna(subset = selected_features)

    # Encoding categorical features into a one-hot encoding
    categorical_features = [
        'side',
        'shot_type', 'team_side', 'period',
        'period_type', 'previous_event_type',
        'current_friendly_on_ice', 'current_opposite_on_ice', 'Rebound'
    ]
    # categorical_features = ['side', 'shot_type', 'period', 'period_type', 'previous_event_type']

    # Ecoding the features
    for feature in categorical_features:
        print(f"Encoding categorical feature {feature}.")
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


    # Split the data into features and labels for train and validation
    x_train, x_valid, y_train, y_valid = train_test_split(data.drop(columns=['is_goal']), data['is_goal'], test_size=0.2, stratify=data['is_goal'])

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

    # normalization/standardization to features
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train[features_standardizing] = scaler.fit_transform(x_train[features_standardizing])
    x_valid[features_standardizing] = scaler.fit_transform(x_valid[features_standardizing])

    # print(data.shape)
    # print(x_train.shape)
    # print(x_valid.shape)
    # print(y_train.shape)
    # print(y_valid.shape)

    return x_train, x_valid, y_train, y_valid, selected_features

#%%
def train_model(x_train, x_valid, y_train, y_valid):

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    opt = keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    epoch = 2

    filepath = 'nn.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    class_weight = {0: 1., 1: 2.}

    # Train the model
    model.fit(x_train, y_train, epochs=epoch, validation_data=(x_valid, y_valid), callbacks=callbacks, class_weight=class_weight)

    # Evaluate the model
    model.evaluate(x_valid, y_valid)

    return model


#%%
def train_nn(x_train, x_valid, y_train, y_valid, features, comet=False):
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
        experiment.log_parameters({'model': 'nn', 'feature': features})


    print('Input shape:', x_train.shape)

    clf = train_model(x_train, x_valid, y_train, y_valid)


    if comet:
        model_name = 'nn'+'_'.join(features)
        joblib.dump(clf, model_name+'.joblib')
        experiment.log_model(model_name, model_name+'.joblib')

    return clf
#%%
def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    sns.set_theme()
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')
#%%
x_train, x_valid, y_train, y_valid, features = prep_data(train_data)

#%%
def main(data_train):

    TOGGLE_TRAIN = True

    x_train, x_valid, y_train, y_valid, features = prep_data(data_train)

    print(x_train.shape)
    print(x_valid.shape)
    print(y_train.shape)
    print(y_valid.shape)


    if TOGGLE_TRAIN:
        clf = train_nn(x_train, x_valid, y_train, y_valid, features, comet=False)


    else:
        # File path
        filepath = './nn.epoch05-loss0.28.hdf5'

        # Load the model
        model = keras.models.load_model(filepath, compile = True)

        # Generate predictions for samples
        predictions = model.predict(x_valid)
        # print(predictions)
        print(np.mean(predictions))
        print(np.std(predictions))


        threshold = 0.28
        predictions[predictions <= threshold] = 0
        predictions[predictions > threshold] = 1

        y_valid = y_valid.to_numpy()


        print(classification_report(y_valid, predictions))
        print(confusion_matrix(y_valid, predictions))

        # correct = 0

        # for i, prediction in enumerate(predictions):
        #     if prediction == y_valid[i]:
        #         correct += 1

        # print(correct / len(predictions))

        # ROC curve
        plot_roc_curve(predictions, y_valid, '-', 'nn distance')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()









if __name__ == "__main__":
    main(train_data)
# %%
