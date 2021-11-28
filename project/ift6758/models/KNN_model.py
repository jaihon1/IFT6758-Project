#%%
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from comet_ml import Experiment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV as GS
from sklearn.ensemble import RandomForestRegressor
import matplotlib.ticker as mtick
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import plot_confusion_matrix

from ift6758.utils.utils import plot_roc_curve, plot_goal_rate, plot_cumulative_sum, plot_calibration



#%%
data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/games_data/games_data_all_seasons.csv')), index_col=0)
# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300]
train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

# Function (borrowed from Jaihon) to have control over desired features
def prep_data(data_train):
    categorical_features = ['side', 'shot_type', 'period', 'period_type', 'previous_event_type', 'Rebound']
    num_features = [
    'coordinate_x', 'coordinate_y',
    'distance_net', 'angle_net', 'time_since_pp_started', 'current_time_seconds', 'current_friendly_on_ice', 'current_opposite_on_ice', 'Change_in_shot_angle',
    'shot_last_event_delta', 'shot_last_event_distance', 'Speed']
    # Set selected features
    selected_features = ['side', 'shot_type',
       'period', 'period_type', 'coordinate_x', 'coordinate_y',
       'is_goal', 'distance_net', 'angle_net', 'previous_event_type',
       'time_since_pp_started', 'current_time_seconds',
       'current_friendly_on_ice', 'current_opposite_on_ice','shot_last_event_delta',
        'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed']

    data = data_train[selected_features]

    # Drop rows with NaN values
    data = data.dropna(subset = selected_features)

    # Encoding categorical features into a one-hot encoding

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
    X_train, X_valid, y_train, y_valid = train_test_split(data.drop(columns=['is_goal']), data['is_goal'], test_size=0.2, stratify=data['is_goal'])
    features_standardizing = num_features

    # normalization/standardization to features
    scaler = StandardScaler()

    X_train[features_standardizing] = scaler.fit_transform(X_train[features_standardizing])
    X_valid[features_standardizing] = scaler.fit_transform(X_valid[features_standardizing])
    y_train = y_train.to_numpy()

    return X_train,X_valid, y_train, y_valid, selected_features


# %%

def train_model(X_train, y_train, selected_features, n_neighbors = [5, 6, 7,8], weights=['uniform', 'distance'], n_estimators=[100,200,300],criterion=['squared_error','poisson'],  comet=False, train_KNN=False, train_forest=False):
    if train_KNN is True:
        tuned_parameters = [
        {"n_neighbors": n_neighbors, "weights": weights}]
        GS_model = GS(KNeighborsRegressor(), tuned_parameters, cv=5, scoring='roc_auc')
        GS_model.fit(X_train, y_train)
        best_params = GS_model.best_params_
        best_params_name = [f'{key} :'+' '+ str(val) for key, val in zip(best_params.keys(), best_params.values())]
        print(best_params_name)
        if comet:
        # Create experiment for comet
            experiment = Experiment(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="ift6758-project",
                workspace="jaihon"
            )
            experiment.log_parameters({'model': 'KNN', 'feature': selected_features.remove('is_goal'), 'Best_params': best_params_name})
        if comet:
            model_name = 'best_KNN_you_ever_seen'
            joblib.dump(GS_model, model_name+'.joblib')
            experiment.log_model(model_name, model_name+'.joblib')
    if train_forest is True:
        tuned_parameters = [
        {"n_estimators": n_estimators, "criterion": criterion}]
        GS_model = GS(RandomForestRegressor(), tuned_parameters, cv=5, scoring='roc_auc')
        GS_model.fit(X_train, y_train)
        best_params = GS_model.best_params_
        best_params_name = [f'{key} :'+' '+ str(val) for key, val in zip(best_params.keys(), best_params.values())]
        print(best_params_name)
        if comet:
        # Create experiment for comet
            experiment = Experiment(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="ift6758-project",
                workspace="jaihon"
            )
            experiment.log_parameters({'model': 'KNN', 'feature': selected_features.remove('is_goal'), 'Best_params': best_params_name})
        if comet:
            model_name = 'best_Random_forest_you_ever_seen'
            joblib.dump(GS_model, model_name+'.joblib')
            experiment.log_model(model_name, model_name+'.joblib')

    return GS_model


def find_optimal_threshold(predictions, true_y):
    scores = []
    threshold = []

    for i in range(0, 100):
        # create a numpy array with the same shape as predictions
        masked_predictions = np.zeros(predictions.shape)
        for j, prediction in enumerate(predictions):
            if prediction <= i/100:
                masked_predictions[j] = 0
            else:
                masked_predictions[j] = 1

        scores.append(f1_score(true_y, masked_predictions))
        threshold.append(i/100)


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
    print(f'Best threshold is : {threshold} for an F1 value of {max(scores)}.')
    return threshold

def main(data_train):

    TOGGLE_TRAIN = False
    TRAIN_FOREST = False
    TRAIN_KNN = False

    X_train,X_valid, y_train, y_valid, selected_features = prep_data(data_train)



    if TOGGLE_TRAIN:
        GS_model = train_model(X_train, y_train, selected_features, comet=True, train_forest=TRAIN_FOREST, train_KNN=TRAIN_KNN)
        if TRAIN_KNN:
            joblib.dump(GS_model, 'KNN_model.pkl')

        if TRAIN_FOREST:
            joblib.dump(GS_model, 'rngforest.pkl')

    else:
        # File path
        filepath_KNN = '../../models/KNN_Final_Final_model.joblib'
        filepath_forest = '../../models/Randomforestclassifier_Final_model.joblib'

        # Load the model
        model_KNN = joblib.load(filepath_KNN , mmap_mode ='r')
        model_forest = joblib.load(filepath_forest , mmap_mode ='r')
        selected_features = ['side', 'shot_type',
       'period', 'period_type', 'coordinate_x', 'coordinate_y',
       'distance_net', 'angle_net', 'previous_event_type',
       'time_since_pp_started', 'current_time_seconds',
       'current_friendly_on_ice', 'current_opposite_on_ice','shot_last_event_delta',
        'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed']

        # Generate predictions for samples
        # predictions_KNN_prob = model_KNN.predict(X_train)
        # predictions_forest_prob = model_forest.predict(X_train)

        # best_KNN_threshold = find_optimal_threshold(predictions_KNN_prob, y_train)
        # best_forest_threshold = find_optimal_threshold(predictions_forest_prob, y_train)
        best_KNN_threshold=0.73
        best_forest_threshold = 0.44
        predictions_KNN_prob = model_KNN.predict(X_valid)
        predictions_forest_prob = model_forest.predict(X_valid)
        predictions_KNN = predictions_KNN_prob.copy()
        predictions_forest = predictions_forest_prob.copy()
        predictions_KNN[predictions_KNN <= best_KNN_threshold] = 0
        predictions_KNN[predictions_KNN > best_KNN_threshold] = 1
        predictions_forest[predictions_forest <= best_forest_threshold] = 0
        predictions_forest[predictions_forest > best_forest_threshold] = 1

        print(confusion_matrix(y_valid, predictions_KNN))
        print(confusion_matrix(y_valid, predictions_forest))

    

        # Goal rate:
        plot_goal_rate([predictions_KNN_prob,predictions_forest_prob], [y_valid, y_valid], ['k-NNRegressor', 'Random Forest'], 'goal_rate')

        #Cumsum
        plot_cumulative_sum([predictions_KNN_prob,predictions_forest_prob], [y_valid, y_valid], ['k-NNRegressor', 'Random Forest'], 'cum_sum')

        #Calibration
        plot_calibration([predictions_KNN_prob, predictions_forest_prob], [y_valid, y_valid], ['k-NNRegressor', 'Random Forest'], 'calibration')

        # ROC curve
        plot_roc_curve([predictions_KNN_prob, predictions_forest_prob], [y_valid, y_valid], ['-', '-'], ['k-NNRegressor', 'Random Forest'], 'roc_curve')
    





if __name__ == "__main__":
    main(train_data)