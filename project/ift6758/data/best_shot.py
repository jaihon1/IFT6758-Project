#%%
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

#%%
# Load the data
data = pd.read_csv("games_data/games_data_all_seasons.csv")


# split into train and test
data['game_pk'] = data['game_pk'].apply(lambda i: str(i))
train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]

#%%
# features = ['game_pk', 'side', 'event_index', 'event_type', 'shot_type',
#     'goal_strength', 'team_id', 'team_side', 'period', 'period_type',
#     'period_time', 'datetime', 'coordinate_x', 'coordinate_y',
#     'player_shooter', 'player_scorer', 'player_goalie', 'empty_net',
#     'is_goal', 'distance_net', 'angle_net', 'previous_event_type',
#     'previous_event_x_coord', 'previous_event_y_coord',
#     'previous_event_period', 'previous_event_period_time',
#     'time_since_pp_started', 'current_time_seconds',
#     'current_friendly_on_ice', 'current_opposite_on_ice'
# ]

def prep_data(data_train):
    # Set seleceted features
    selected_features = ['side', 'shot_type',
       'period', 'period_type', 'coordinate_x', 'coordinate_y', 'empty_net',
       'is_goal', 'distance_net', 'angle_net', 'previous_event_type',
       'time_since_pp_started', 'current_time_seconds',
       'current_friendly_on_ice', 'current_opposite_on_ice'
    ]

    data = data_train[selected_features]

    # Drop rows with NaN values
    data = data.dropna(subset = selected_features)

    # Encoding categorical features into a one-hot encoding
    categorical_features = ['side', 'shot_type', 'period', 'period_type', 'empty_net', 'previous_event_type']

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
        # print("encoding:")
        # print(data.shape)
        # print(df_encoded.shape)


    # Split the data into features and labels for train and validation
    x_train, x_valid, y_train, y_valid = train_test_split(data.drop(columns=['is_goal']), data['is_goal'], test_size=0.2, stratify=data['is_goal'])

    # print(data.shape)
    # print(x_train.shape)
    # print(x_valid.shape)
    # print(y_train.shape)
    # print(y_valid.shape)

    return x_train, x_valid, y_train, y_valid

#%%
def train(x_train, x_valid, y_train, y_valid):

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    epoch = 7

    filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    # Train the model
    model.fit(x_train, y_train, epochs=epoch, validation_data=(x_valid, y_valid), callbacks=callbacks)

    # Evaluate the model
    model.evaluate(x_valid, y_valid)


#%%

def main(data_train):
    x_train, x_valid, y_train, y_valid = prep_data(data_train)

    # train(x_train, x_valid, y_train, y_valid)



    # File path
    filepath = './my_best_model.epoch07-loss0.27.hdf5'

    # Load the model
    model = keras.models.load_model(filepath, compile = True)

    # Generate predictions for samples
    predictions = model.predict(x_valid)
    # print(predictions)
    # print(np.mean(predictions))
    # print(np.std(predictions))


    threshold = 0.26
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








if __name__ == "__main__":
    main(train)
# %%
