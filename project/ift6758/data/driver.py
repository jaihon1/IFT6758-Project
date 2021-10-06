from GamesInfo import GamesInfo
from EventGenerator import EventGenerator
import pandas as pd
import json
import os




def main():
    '''Main function that uses json files from games_data directory and points to EventGenerator class to generate 
    a single dataframe containing all desired features from liveData, from all games for seasons ranging from 2016 to 2020.
    
    Returns: Dataframe'''
    seasons = [2016]
    GamesInfo(seasons)


    data = None
    dataframe = pd.DataFrame()
    dirpath = os.path.join(os.path.dirname(__file__))
    dirpath_games_data = os.path.join(dirpath, 'games_data')
    for season in seasons:
        for game_filepath in os.listdir(os.path.join(dirpath_games_data, str(season))):
            with open(os.path.join(dirpath_games_data, str(season), game_filepath)) as file:
                print(f'Creating  dataframe for {game_filepath} file...')
                data = json.load(file)
            live_events = data['liveData']['plays']['allPlays']
            game_pk = data['gamePk']
            game = EventGenerator(game_pk, live_events)
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

