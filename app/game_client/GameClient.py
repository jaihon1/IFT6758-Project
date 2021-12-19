import requests
import json
import logging

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

from EventGenerator import EventGenerator


logger = logging.getLogger(__name__)


class GameClient: 
    def __init__(self):
        self.game_id = None
        self.features = ['side', 'shot_type', 'period', 'period_type', 'coordinate_x', 'coordinate_y', 'is_goal',
                         'team_side', 'distance_net', 'angle_net', 'previous_event_type', 'time_since_pp_started',
                         'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                         'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                         'shot_last_event_delta', 'shot_last_event_distance', 'Rebound', 'Change_in_shot_angle', 'Speed'
                        ]

        self.categorical_features = ['side', 'shot_type', 'period', 'period_type', 'team_side', 'previous_event_type']

    def ping_game(self,game_id):
        # Generate and save dataframe about specific features
        data = self.__download(game_id)
        if data is None:
            logger.error("Unable to get the game data. Make sure the Game ID exists.")
            return pd.DataFrame()
        logger.log("Game data successfully downloaded.")
        #file_path = '/home/johannplantin4/Documents/Dev/Code/IFT6758_group_project/IFT6758-Project/project/ift6758/data/games_data/2019/2019020900.json'
        # with open(file_path, 'r') as jaihon:
        #     data = json.load(jaihon)
        if self.game_id != game_id:
            self.idx = 0
            self.game_pk = data['gamePk']
            self.home = data['gameData']['teams']['home']['triCode']
            self.away = data['gameData']['teams']['away']['triCode']
            self.game_type = data['gameData']['game']['type']
        sides = dict()

        for period in data['liveData']['linescore']['periods']:
            sides[period['num']] = {self.home: period['home'].setdefault('rinkSide', np.NaN), self.away: period['away'].setdefault('rinkSide', np.NaN)}
        live_events = data['liveData']['plays']['allPlays']
        # if self.idx == 0:
        #     live_events = live_events[:len(live_events)//4]
        # else:
        #     live_events = live_events[:self.idx +len(live_events)//4]
        if self.game_id != game_id:
            self.game = EventGenerator(self.game_pk, self.home, self.away, sides, live_events, self.game_type)
            self.game_id = game_id
        else:
            self.game.live_events = live_events
            self.game.sides = sides
        dataframe = self.game.build(self.idx)
        self.idx = len(self.game.live_events)

        if len(dataframe) != 0:
            final_df = self.__add_new_features(dataframe)
            final_df = final_df[self.features]

            # Drop rows with NaN values
            final_df = final_df.dropna(subset=self.features)
            final_df = self.__encode_categorical(final_df)
        else:
            final_df = pd.DataFrame()

        return final_df

    def __download(self, game_id):
        print('Downloading game:', game_id)

        # get the general information of all the regular and playoff games from the season
        url_game = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'

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
        for feature in self.categorical_features:
            one_hot_encoder = OneHotEncoder(sparse=False)
            encoding_df = data[[feature]]

            one_hot_encoder.fit(encoding_df)

            df_encoded = one_hot_encoder.transform(encoding_df)

            df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)

            # Drop original feature and add encoded features
            data.drop(columns=[feature], inplace=True)

            data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

        return data
