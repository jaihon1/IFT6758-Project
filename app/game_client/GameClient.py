from EventGenerator import EventGenerator
import numpy as np
import pandas as pd
import requests
from feat_eng2 import add_new_features
import json


class GameClient: 
    def __init__(self):
        self.game_id = None
    def ping_game(self,game_id):
        # Generate and save dataframe about specific features
        data = self.__download(game_id)
        #file_path = '/home/johannplantin4/Documents/Dev/Code/IFT6758_group_project/IFT6758-Project/project/ift6758/data/games_data/2019/2019020900.json'
        # with open(file_path, 'r') as jaihon:
        #     data = json.load(jaihon)
        if self.game_id != game_id:
            self.idx = 0
            self.game_pk = data['gamePk']
            self.home = data['gameData']['teams']['home']['triCode']
            self.away = data['gameData']['teams']['away']['triCode']
            self.game_type =  data['gameData']['game']['type']
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
            final_df = add_new_features(dataframe)
        else:
            final_df = pd.DataFrame()

        return final_df

    def __download(self, game_id):
        print('Downloading game:', game_id)

        # get the general information of all the regular and playoff games from the season
        url_game = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'

        # iterate over the games included in ids to get the whole information on a game with its ID
        ids = requests.get(url_game).json()

        return ids
