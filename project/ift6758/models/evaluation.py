#%%
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from ift6758.utils.utils import plot_roc_curve, plot_calibration, plot_cumulative_sum, plot_goal_rate

from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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

    elif model == 'logreg':
        # Set seleceted features
        selected_features = [
            'game_pk',
            'is_goal',
            'distance_net', 'angle_net'
        ]
        data = data_raw[selected_features]

        # Drop rows with NaN values
        data = data.dropna(subset = selected_features)

        # Encoding categorical features into a one-hot encoding
        categorical_features = []

        features_standardizing = [
            'distance_net', 'angle_net'
        ]

    elif model == 'knn' or model == 'rndf':
        # Set seleceted features
        selected_features = [
            'game_pk',
            'side', 'shot_type',
            'period', 'period_type', 'coordinate_x', 'coordinate_y',
            'is_goal', 'distance_net', 'angle_net', 'previous_event_type',
            'time_since_pp_started', 'current_time_seconds',
            'current_friendly_on_ice', 'current_opposite_on_ice','shot_last_event_delta',
            'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
        ]
        data = data_raw[selected_features]

        # Drop rows with NaN values
        data = data.dropna(subset = selected_features)

        # Encoding categorical features into a one-hot encoding
        categorical_features = ['side', 'shot_type', 'period', 'period_type', 'previous_event_type', 'Rebound']

        features_standardizing = [
            'coordinate_x', 'coordinate_y',
            'distance_net', 'angle_net', 'time_since_pp_started', 'current_time_seconds', 'current_friendly_on_ice', 'current_opposite_on_ice', 'Change_in_shot_angle',
            'shot_last_event_delta', 'shot_last_event_distance', 'Speed'
        ]

    elif model == 'xgboost':
        # Set seleceted features
        selected_features = [
            'game_pk',
            'side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'is_goal',
            'team_side', 'distance_net', 'angle_net', 'previous_event_type', 'time_since_pp_started',
            'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
            'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
            'shot_last_event_delta', 'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
        ]
        data = data_raw[selected_features]

        # Drop rows with NaN values
        data = data.dropna(subset = selected_features)

        # Encoding categorical features into a one-hot encoding
        categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side', 'previous_event_type']

        features_standardizing = [
            'coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'time_since_pp_started',
            'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
            'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
            'shot_last_event_delta', 'shot_last_event_distance', 'Change_in_shot_angle', 'Speed'
        ]

    else:
        raise ValueError('Model not supported')


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


    # normalization/standardization to features
    if std:
        scaler = StandardScaler()
        data[features_standardizing] = scaler.fit_transform(data[features_standardizing])

    return data


# %%

def prediction_report(predictions, target, threshold=0.5):
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

# %%
PLAYOFF_TOGGLE = False

# %%
'''
NEURAL NETWORK MODELS
'''
# Load the data
if PLAYOFF_TOGGLE:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons_full.csv")

    # pandas replace all values in a column period with a 4 where period = 5, 6, 7, 8
    data['period'] = data['period'].replace({6: 4, 7: 4, 8: 4})

    period_type = 'P'

    # Select data period type by
    data = data[data['game_type'] == period_type]
    data.drop(columns=['game_type'], inplace=True)

else:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

x_test_NN, y_test_NN = prepare(data, bonus=True, model_type='nn', std=True)
x_test_nobonus_NN, y_test_nobonus_NN = prepare(data, bonus=False, model_type='nn', std=True)


# Load the model
model = keras.models.load_model('models/nn/best_shot_nn_final.hdf5', compile = True)
model1 = keras.models.load_model('models/nn/neuralnet_no_dropout.hdf5', compile = True)
model2 = keras.models.load_model('models/nn/neuralnet_nobonus.hdf5', compile = True)

# Generate predictions for samples
predictions_NN = model.predict(x_test_NN)
predictions_NN_unlisted =[i[0] for i in predictions_NN]


print('Printing prediction report for Neural Network model...')
prediction_report(predictions_NN, y_test_NN, threshold=0.33)

predictions_NN_unlisted_int = predictions_NN.copy()
predictions_NN_unlisted_int[predictions_NN_unlisted_int <= 0.33] = 0
predictions_NN_unlisted_int[predictions_NN_unlisted_int > 0.33] = 1
confusion_matrix_NN = confusion_matrix(y_test_NN, predictions_NN_unlisted_int)

# %%
'''
BASELINE MODELS
'''
# Load the data
if PLAYOFF_TOGGLE:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons_full.csv")

    # pandas replace all values in a column period with a 4 where period = 5, 6, 7, 8
    data['period'] = data['period'].replace({6: 4, 7: 4, 8: 4})

    period_type = 'P'

    # Select data period type by
    data = data[data['game_type'] == period_type]
    data.drop(columns=['game_type'], inplace=True)

else:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

x_test_reg, y_test_reg = prepare(data, bonus=True, model_type='logreg', std=True)

# Load the model
model_angle = joblib.load('models/baseline/regression_angle_net.joblib')
model_distance = joblib.load('models/baseline/regression_distance_net.joblib')
model_angle_distance = joblib.load('models/baseline/regression_distance_net_angle_net.joblib')

# Generate predictions for samples: Returns the probability of class_1
predictions_angle_net = model_angle.predict_proba(x_test_reg[['angle_net']].abs())[:, 1]
predictions_distance = model_distance.predict_proba(x_test_reg[['distance_net']])[:, 1]
predictions_angle_distance = model_angle_distance.predict_proba(x_test_reg[['angle_net', 'distance_net']].abs())[:, 1]

# Generate report and curves
print('Printing prediction report for logistic regressor models...')
prediction_report(predictions_angle_net, y_test_reg)
prediction_report(predictions_distance, y_test_reg)
prediction_report(predictions_angle_distance, y_test_reg)
predictions_angle_net_int = predictions_angle_net.copy()
predictions_angle_net_int[predictions_angle_net_int <= 0.5] = 0
predictions_angle_net_int[predictions_angle_net_int > 0.5] = 1

predictions_distance_int = predictions_distance.copy()
predictions_distance_int[predictions_distance_int <= 0.5] = 0
predictions_distance_int[predictions_distance_int > 0.5] = 1

predictions_angle_distance_int = predictions_angle_distance.copy()
predictions_angle_distance_int[predictions_angle_distance_int <= 0.5] = 0
predictions_angle_distance_int[predictions_angle_distance_int > 0.5] = 1

confusion_matrix_reg_1 = confusion_matrix(y_test_reg, predictions_angle_net_int)
confusion_matrix_reg_2 = confusion_matrix(y_test_reg, predictions_distance_int)
confusion_matrix_reg_3 = confusion_matrix(y_test_reg, predictions_angle_distance_int)


# %%
'''
XGBOOST MODELS
'''
# Load the data
if PLAYOFF_TOGGLE:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons_full.csv")

    # pandas replace all values in a column period with a 4 where period = 5, 6, 7, 8
    data['period'] = data['period'].replace({6: 4, 7: 4, 8: 4})

    period_type = 'P'

    # Select data period type by
    data = data[data['game_type'] == period_type]
    data.drop(columns=['game_type'], inplace=True)

else:
    data = pd.read_csv("ift6758/data/games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300] # remove outliers with value = inf

x_test_XG, y_test_XG = prepare(data, bonus=True, model_type='xgboost', std=True)

# Load the model
model = joblib.load('models/xgb_tuning_best.joblib')

# Generate predictions for samples: Returns the probability of class_1
predictions_xg = model.predict_proba(x_test_XG)[:, 1]

# Generate report and curves
print('Printing prediction report xgboost model...')
prediction_report(predictions_xg, y_test_XG)
predictions_xg_int = predictions_xg.copy()
predictions_xg_int[predictions_xg_int <= 0.5] = 0
predictions_xg_int[predictions_xg_int > 0.5] = 1
confusion_matrix_xg = confusion_matrix(y_test_XG, predictions_xg_int)

#Generating curves for
# Goal rate:
plot_goal_rate([predictions_NN_unlisted, predictions_angle_distance, predictions_angle_net, predictions_distance, predictions_xg], [y_test_NN, y_test_reg, y_test_reg, y_test_reg, y_test_XG], ['NeuralNetwork', 'LogisticRegression 1', 'LogisticRegression 2', 'LogisticRegression 3', 'XGBoost' ])

#Cumsum
plot_cumulative_sum([predictions_NN_unlisted, predictions_angle_distance, predictions_angle_net, predictions_distance, predictions_xg], [y_test_NN, y_test_reg, y_test_reg, y_test_reg, y_test_XG], ['NeuralNetwork', 'LogisticRegression 1', 'LogisticRegression 2', 'LogisticRegression 3', 'XGBoost' ])

#Calibration
plot_calibration([predictions_NN_unlisted, predictions_angle_distance, predictions_angle_net, predictions_distance, predictions_xg], [y_test_NN, y_test_reg, y_test_reg, y_test_reg, y_test_XG], ['NeuralNetwork', 'LogisticRegression 1', 'LogisticRegression 2', 'LogisticRegression 3', 'XGBoost' ])
# ROC curve
plot_roc_curve([predictions_NN_unlisted, predictions_angle_distance, predictions_angle_net, predictions_distance, predictions_xg], [y_test_NN, y_test_reg, y_test_reg, y_test_reg, y_test_XG],['-', '-','-', '-','-'], ['NeuralNetwork', 'LogisticRegression 1', 'LogisticRegression 2', 'LogisticRegression 3', 'XGBoost' ])

cm_1 = pd.DataFrame(confusion_matrix_NN, index=[0,1], columns=[0,1])

cm_2 = pd.DataFrame(confusion_matrix_xg, index=[0,1], columns=[0,1])

cm_3 = pd.DataFrame(confusion_matrix_reg_1, index=[0,1], columns=[0,1])

cm_4 = pd.DataFrame(confusion_matrix_reg_2, index=[0,1], columns=[0,1])

cm_5 = pd.DataFrame(confusion_matrix_reg_3, index=[0,1], columns=[0,1])

cm= [cm_1, cm_2, cm_3, cm_4, cm_5]
titel = ['Neural Network', 'Logistic Regression 1', 'Logistic Regression 2', 'Logistic Regression 3', 'XG Boost']
iterator=0
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))
for matrice, ax in zip(cm, axes.flatten()):
    sns.heatmap(matrice, annot=True, cmap=sns.color_palette("light:b", as_cmap=True), fmt='g', ax=ax)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')
    ax.title.set_text(titel[iterator])
    iterator += 1
fig.delaxes(axes[2][1])
plt.tight_layout()
plt.show()