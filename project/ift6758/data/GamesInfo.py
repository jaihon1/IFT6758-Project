from glob import glob
import requests
import json
import os



class GamesInfo:
    def __init__(self, season, filepath):
        self.season = season
        self.filepath = filepath


    def load(self, season, filepath):
        data = []
        for file_name in glob('*.json'):
            with open(filepath) as f:
                data.append(json.load(f))


        """
        Should load the json files for the corresponding season located at filepath.
        Args:
            season:
            filepath:
        Returns:
        """


    def download(self, season, filepath):
        """
        Download and save the json files from corresponding season in the specified filepath.
        Args:
            filepath:
            season:

        Returns:

        """
        # create the format for season. ex: if season == 2017, we need to put it into the format 20172018
        season = str(season) + str(int(season) + 1)

        # get the general information of all the regular and playoff games from the season
        url_season = 'https://statsapi.web.nhl.com/api/v1/schedule?season=' + season
        url_regular = url_season + '&gameType=R'
        url_playoff = url_season + '&gameType=P'
        urls = [url_regular, url_playoff]

        for url in urls:
            ids = requests.get(url).json()

            # iterate over the games included in regular_ids to get the whole information on a game with its ID
            for date in ids['dates']:
                for game in date['games']:

                    # get the json file of the game from the api
                    game_url = game['link']
                    game_info = requests.get('https://statsapi.web.nhl.com' + game_url).json()

                    # Create the directory where to save file
                    directory = os.path.join(filepath, season, game['gameType'])
                    if not os.path.isdir(filepath):
                        os.makedirs(filepath)

                    # save file
                    f_path_save = os.path.join(directory, str(game['gamePk']))
                    with open(f_path_save, 'w') as file:
                        json.dump(game_info, file)
