import numpy as np

from GamesInfo import GamesInfo
from EventGenerator import EventGenerator
import pandas as pd
import json
import os




def main():
    '''Main function that uses json files from games_data directory and points to EventGenerator class to generate 
    a single dataframe containing all desired features from liveData, from all games for seasons ranging from 2016 to 2020.
    
    Returns: Dataframe'''
    seasons = [2016, 2017, 2018, 2019, 2020]

    dirpath = os.path.join(os.path.dirname(__file__))

    all_games = GamesInfo(seasons, dirpath)


    data = None
    dataframe = pd.DataFrame()
    dirpath_games_data = os.path.join(dirpath, 'games_data')
    for season in seasons:
        for data in all_games.all_games[season]:
            live_events = data['liveData']['plays']['allPlays']
            game_pk = data['gamePk']
            home = data['gameData']['teams']['home']['triCode']
            away = data['gameData']['teams']['away']['triCode']
            sides = dict()
            for period in data['liveData']['linescore']['periods']:
                sides[period['num']] = {home: period['home'].setdefault('rinkSide', np.NaN), away: period['away'].setdefault('rinkSide', np.NaN)}
            game = EventGenerator(game_pk, home, away, sides, live_events)
            if len(dataframe) == 0:
                dataframe = game.build()
            else:
                temp_df = game.build()
                dataframe = pd.concat([dataframe, temp_df])
            print(len(dataframe))
    print(dataframe.iloc[:11,:])
    dataframe.to_csv(f'{dirpath_games_data}/games_data_all_seasons.csv')
    return dataframe



if __name__ == "__main__":
    main()

