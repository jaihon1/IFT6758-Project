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
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#%%
# Load the data
# data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")
data = pd.read_csv("games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

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


    # Ecoding the features
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

    return x_train, x_valid, y_train, y_valid, selected_features

#%%
# Compute percentile
def plot_goal_rate(proba, true_y, label):
    percentile = np.arange(0, 100, 2)
    percentile_pred = np.percentile(proba, percentile)

    y_valid_df = pd.DataFrame(true_y)

    y_valid_df['bins_percentile'] = pd.cut(proba, percentile_pred, duplicates='drop')
    goal_rate_by_percentile = y_valid_df.groupby(by=['bins_percentile']).apply(lambda g: g['is_goal'].sum()/len(g))

    if len(percentile_pred)-len(goal_rate_by_percentile) == 1:
        percentile = percentile[:-1]
    else:
        percentile = percentile[:-2]

    sns.set_theme()
    g = sns.lineplot(x=percentile, y=goal_rate_by_percentile*100, label=label)
    ax = g.axes
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    return y_valid_df

#%%
# plot cumulative proportion of goals as function of shot probability model percentile
def plot_cumulative_sum(true_y, y_valid_df, label):
    percentile = np.arange(0,100,2)

    total_number_goal = (true_y == 1).sum()

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


#%%
def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    sns.set_theme()
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')


def find_optimal_threshold(predictions, true_y):
    scores = []

    for i in range(0, 100):
        # create a numpy array with the same shape as predictions
        masked_predictions = np.zeros(predictions.shape)
        for j, prediction in enumerate(predictions):
            if prediction <= i/100:
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

#%%
def train_model(x_train, x_valid, y_train, y_valid, class_weight, epoch, lr):

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    filepath = 'nn.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    # Train the model
    model.fit(x_train, y_train, epochs=epoch, validation_data=(x_valid, y_valid), callbacks=callbacks, class_weight=class_weight, batch_size=128)

    # Evaluate the model
    model.evaluate(x_valid, y_valid)

    return model


#%%
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


    print('Input shape:', x_train.shape)

    clf = train_model(x_train, x_valid, y_train, y_valid, class_weight, epoch=30, lr=0.001)


    if comet:
        model_name = 'nn'+'_'
        joblib.dump(clf, model_name+'.joblib')
        experiment.log_model(model_name, model_name+'.joblib')

    return clf

#%%
def main(data_train):

    TOGGLE_TRAIN = False

    print(os.environ.get("COMET_API_KEY"))

    x_train, x_valid, y_train, y_valid, features = prep_data(data_train)



    if TOGGLE_TRAIN:
        clf = train_nn(x_train, x_valid, y_train, y_valid, features, comet=True)

    else:
        # File path
        filepath = 'nn_models/nn.epoch41-loss0.33.hdf5'

        # Load the model
        model = keras.models.load_model(filepath, compile = True)

        # Generate predictions for samples
        predictions = model.predict(x_valid)


        find_optimal_threshold(predictions, y_valid)

        # # ROC curve
        # plot_roc_curve(predictions, y_valid.to_numpy(), '-', 'nn distance')
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.legend()
        # plt.show()

        # # Goal rate
        # valid_goal_rate = plot_goal_rate(predictions.flatten(), y_valid, 'NN')

        # plt.xlim(100, 0)
        # plt.ylim(0, 100)
        # plt.xlabel('Shot probability model percentile')
        # plt.ylabel('Goals / (Shots + Goals)')
        # plt.show()


        # # Cumulative goal rate
        # plot_cumulative_sum(y_valid, valid_goal_rate, 'NN')

        # plt.xlim(100, 0)
        # plt.ylim(0, 100)
        # plt.xlabel('Shot probability model percentile')
        # plt.ylabel('Proportion')
        # plt.show()


        # calibration curve
        sns.set_theme()
        fig = plt.figure()
        ax = plt.axes()
        disp_random = CalibrationDisplay.from_predictions(y_valid, predictions, n_bins=25, ax=ax, name='Random classifier', ref_line=False)
        plt.xlim(0,0.3)
        plt.legend(loc=2)
        plt.show()









if __name__ == "__main__":
    main(train_data)
# %%
