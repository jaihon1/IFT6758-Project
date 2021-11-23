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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import GridSearchCV as GS
from sklearn.ensemble import RandomForestRegressor


#%%
data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/games_data/games_data_all_seasons.csv')),index_col=0)
# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
data = data[data['Speed'] < 300]
train_data, test_data = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

# Function (borrowed from Jaihon) to have control over desired features
def prep_data(data_train):
    feature_list = [i for i in train_data.columns]
    categorical_features = ['side', 'shot_type', 'period', 'period_type', 'previous_event_type', 'Rebound']
    num_features = [
    'coordinate_x', 'coordinate_y',
    'distance_net', 'angle_net', 'time_since_pp_started', 'current_time_seconds', 'current_friendly_on_ice', 'current_opposite_on_ice', 'Change_in_shot_angle',
    'shot_last_event_delta', 'shot_last_event_distance', 'Speed']
    # Set seleceted features
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
    #print(X_train, X_train.shape, y_train, y_train.shape)
    features_standardizing = num_features

    # normalization/standardization to features
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    # for i in X_train[features_standardizing]:
    #     X_train[i] = X_train[i].astype(float)
    X_train[features_standardizing] = scaler.fit_transform(X_train[features_standardizing])
    X_valid[features_standardizing] = scaler.fit_transform(X_valid[features_standardizing])
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()

    return X_train,X_valid, y_train, y_valid, selected_features


# %%

def train_model(X_train, y_train, n_neighbors = [5, 6, 7,8], weights=['uniform', 'distance'], n_estimators=[100,200,300],criterion=['squared_error', 'absolute_error', 'poisson'],  comet=False, train_KNN=False, train_forest=False):
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

def plot_roc_curve(pred_prob, true_y, marker, label):
    score = roc_auc_score(true_y, pred_prob)
    fpr, tpr, _ = roc_curve(true_y, pred_prob)
    sns.set_theme()
    plt.grid(True)
    plt.plot(fpr, tpr, linestyle=marker, label=label+f' (area={score:.2f})')

X_train,X_valid, y_train, y_valid, selected_features = prep_data(train_data)
def main(data_train):

    TOGGLE_TRAIN = True
    THRESHOLD_TWEAKING = True
    TRAIN_FOREST = False
    TRAIN_KNN = True

    X_train,X_valid, y_train, y_valid, selected_features = prep_data(data_train)



    if TOGGLE_TRAIN:
        GS_model = train_model(X_train, y_train, comet=False, train_forest=TRAIN_FOREST, train_KNN=TRAIN_KNN)
        predictions = GS_model.predict(X_valid)
        if THRESHOLD_TWEAKING:
            for i in np.linspace(0.1,1,10):
                temp_predictions = GS_model.predict(X_valid)
                threshold = i
                temp_predictions[temp_predictions <= threshold] = 0
                temp_predictions[temp_predictions > threshold] = 1
                print(f'Displaying scores for threshold of {i}')
                print(classification_report(y_valid, temp_predictions))
                print(roc_auc_score(y_valid, temp_predictions))
        best_KNN_threshold = None
        best_Forest_threshold = None
        if TRAIN_KNN:
            threshold = best_KNN_threshold
            predictions[predictions <= threshold] = 0
            predictions[predictions > threshold] = 1      
            print(predictions)
            print(classification_report(y_valid, predictions))
            conf_matrix = confusion_matrix(y_valid, predictions)
        if TRAIN_FOREST:
            threshold = best_Forest_threshold
            predictions[predictions <= threshold] = 0
            predictions[predictions > threshold] = 1      
            print(predictions)
            print(classification_report(y_valid, predictions))
            conf_matrix = confusion_matrix(y_valid, predictions)                

        # ROC curve
        if TRAIN_KNN:
            plot_roc_curve(predictions, y_valid, '-', 'KNN distance')
        if TRAIN_FOREST:
            plot_roc_curve(predictions, y_valid, '-', 'Forest distance')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()


    else:
        pass
        # File path
        #filepath = './nn.epoch05-loss0.28.hdf5'

        # Load the model
        #model = keras.models.load_model(filepath, compile = True)

        # Generate predictions for samples
        #predictions = model.predict(x_valid)
        # print(predictions)
        # print(np.mean(predictions))
        # print(np.std(predictions))


        # threshold = 0.28
        # predictions[predictions <= threshold] = 0
        # predictions[predictions > threshold] = 1

        # y_valid = y_valid.to_numpy()


        # print(classification_report(y_valid, predictions))
        # print(confusion_matrix(y_valid, predictions))

        # # correct = 0

        # # for i, prediction in enumerate(predictions):
        # #     if prediction == y_valid[i]:
        # #         correct += 1

        # # print(correct / len(predictions))


if __name__ == "__main__":
    main(train_data)