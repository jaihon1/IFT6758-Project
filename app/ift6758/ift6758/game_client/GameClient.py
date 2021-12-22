import requests
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from .EventGenerator import EventGenerator

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class GameClient:
    def __init__(self):
        self.game_id = None
        self.game_pk = None
        self.home = None
        self.away = None
        self.game_type = None
        self.dateTime = None
        self.period = 0
        self.time_left = "20:00"
        self.features = ['side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'is_goal',
                         'team_side', 'distance_net', 'angle_net', 'previous_event_type', 'time_since_pp_started',
                         'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                         'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                         'shot_last_event_delta', 'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
                        ]

        self.categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side', 'previous_event_type']
        self.categorical_features_encoded = [
            ['away', 'home'],
            ['Backhand', 'Deflected', 'Snap Shot', 'Slap Shot', 'Wrap-around', 'Wrist Shot', 'Tip-In'],
            [1, 2, 3, 4],
            ['OVERTIME', 'REGULAR'],
            ['left', 'right'],
            ['BLOCKED_SHOT', 'GOAL', 'MISSED_SHOT', 'SHOT', 'FACEOFF', 'GIVEAWAY', 'HIT', 'PENALTY', 'TAKEAWAY']
        ]

    def ping_game(self, game_id):
        # Generate and save dataframe about specific features
        data = self.__download(game_id)
        if data is None:
            return pd.DataFrame()
        if self.game_id != str(game_id):
            self.idx = 0
            self.game_pk = data['gamePk']
            self.home = data['gameData']['teams']['home']['triCode']
            self.away = data['gameData']['teams']['away']['triCode']
            self.game_type = data['gameData']['game']['type']
            self.dateTime = data['gameData']['datetime']['dateTime']

        sides = dict()
        for period in data['liveData']['linescore']['periods']:
            sides[period['num']] = {self.home: period['home'].setdefault('rinkSide', 'right'), self.away: period['away'].setdefault('rinkSide', 'left')}

        live_events = data['liveData']['plays']['allPlays']

        if self.game_id != str(game_id):
            self.game = EventGenerator(self.game_pk, self.home, self.away, sides, live_events, self.game_type)
            self.game_id = str(game_id)

        self.game.live_events = live_events
        self.game.sides = sides

        dataframe = self.game.build(self.idx)
        self.idx = len(self.game.live_events)
        if len(dataframe) != 0:
            self.time_left = self.game.live_events[-1]['about']['periodTimeRemaining']
            self.period = self.game.live_events[-1]['about']['period']
            final_df = self.__add_new_features(dataframe)
            final_df = final_df[self.features]

            # Drop rows with NaN values
            final_df = final_df.dropna(subset=self.features)
            final_df = self.__encode_categorical(final_df)

            final_df['Rebound'] = final_df['Rebound'].astype('bool')
        else:
            final_df = pd.DataFrame()

        return final_df

    def __download(self, game_id):

        # get the general information of all the regular and playoff games from the season
        url_game = f'https://statsapi.web.nhl.com/api/v1/game/{int(game_id)}/feed/live/'

        r = requests.get(url_game)

        if r.status_code != 200:
            return None

        return r.json()

    def __add_new_features(self, dataframe):
        df_filtered = dataframe
        df_filtered.drop(columns=['period_time','previous_event_period_time'], inplace=True)
        df_filtered['shot_last_event_delta'] = [t1 - t0 for t1, t0 in zip(df_filtered['current_time_seconds'], df_filtered['previous_event_time_seconds'])]
        df_filtered['shot_last_event_distance'] = [np.sqrt((x2-x1)**2 + (y2 - y1)**2) for x1, x2, y1, y2 in zip(df_filtered['previous_event_x_coord'], df_filtered['coordinate_x'], df_filtered['previous_event_y_coord'], df_filtered['coordinate_y'])]

        ### Adding the Rebound Feature
        df_filtered['Rebound'] = ['True' if all((last_event_type == 'SHOT', last_event_team == event_team, last_period == period)) else 'False' for last_event_type, last_event_team, event_team, last_period, period in zip(df_filtered['previous_event_type'], df_filtered['previous_event_team'], df_filtered['team_id'], df_filtered['previous_event_period'], df_filtered['period'])]

        # ### Adding Change in shot angle feature
        myradians = np.arctan2(df_filtered['previous_event_y_coord']-df_filtered['coordinate_y'], df_filtered['previous_event_x_coord']-df_filtered['coordinate_x'])
        df_filtered.loc[df_filtered['Rebound'] == 'True', 'Change_in_shot_angle'] = np.degrees(myradians)
        df_filtered.loc[df_filtered['Rebound'] == 'False', 'Change_in_shot_angle'] = '0'

        # ### Adding Speed feature: distance from the previous event, divided by the time since the previous event
        df_filtered['Speed'] = df_filtered['shot_last_event_distance'] / df_filtered['shot_last_event_delta']
        df_filtered.drop(columns=df_filtered.columns[0], inplace=True)
        max_speed = df_filtered['Speed'].replace([np.inf], -np.inf).max()
        df_filtered['Speed'] = df_filtered['Speed'].replace([np.inf], max_speed)

        return df_filtered

    def __encode_categorical(self, data):
        # Encoding categorical features into a one-hot encoding
        for feature, encoding in zip(self.categorical_features, self.categorical_features_encoded):
            one_hot_encoder = OneHotEncoder(sparse=False, categories=[encoding])
            encoding_df = data[[feature]]

            one_hot_encoder.fit(encoding_df)

            df_encoded = one_hot_encoder.transform(encoding_df)

            df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)

            # Drop original feature and add encoded features
            data.drop(columns=[feature], inplace=True)

            data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

        return data
