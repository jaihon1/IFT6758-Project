from glob import glob
import requests
import json
import os


class GamesInfo:

    def __init__(self, season, filepath=os.path.join(os.path.dirname(__file__)), games_info=None):
        """

        Generate a GamesInfo object which contains all of the regular and playoff games from the specified
        seasons.

        Args:
            season: int, string or list of int or string, the season from which we want to load all of the games
            filepath: the directory where the downloaded files are or where to download the files.
            games_info: dictionary of all the games information from all the seasons specified by season.
        """
        # we want the attribute season to be a list of ints
        if type(season) is int or type(season) is str:
            self.season = [int(season)]
        elif type(season) is list or type(season) is tuple:
            self.season = [int(s) for s in season]
        self.filepath = os.path.join(filepath, 'games_data')

        if games_info is None:
            self.all_games = dict()
            for s in self.season:
                # if folder containing the season exists, then check if folder is empty, else download
                if os.path.isdir(os.path.join(self.filepath, str(s))):
                    # if folder is empty, then download, else load
                    if len(os.listdir(self.filepath)) == 0:
                        self.all_games[s] = self.__download(s)
                    else:
                        self.all_games[s] = self.__load(s)
                else:
                    self.all_games[s] = self.__download(s)
        else:
            self.all_games = games_info

    def __load(self, season):
        """

        Should load the json files in self.all_games for the corresponding seasons located at filepath.

        Args:
            season: int or string of the name of the season to add, ex: 2017 for the 20172018 season
        Returns: a list of all the information about all the games of the season. The length of the list is
        the number of games in the season.

        """
        directory = os.path.join(self.filepath, str(season))

        print('Loading games from', season, 'located at', directory)

        games_info = []
        for file_name in glob(os.path.join(directory, '*.json')):
            with open(file_name) as f:
                games_info.append(json.load(f))
        return games_info

    def __download(self, season):
        """

        Download and save the json files from corresponding seasons in the specified filepath.

        Args:
            season: int or string of the name of the season to add, ex: 2017 for the 20172018 season
        Returns: a list of all the information about all the games of the season. The length of the list is
        the number of games in the season.

        """
        print('Downloading games from season:', season)

        # create the format for season. ex: if season == 2017, we need to put it into the format 20172018
        season_name = str(season) + str(int(season) + 1)
        games_info = []
        # get the general information of all the regular and playoff games from the season
        url_season = 'https://statsapi.web.nhl.com/api/v1/schedule?season=' + season_name + '&gameType=R&gameType=P'

        # Create the directory where to save file
        directory = os.path.join(self.filepath, str(season))
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # iterate over the games included in ids to get the whole information on a game with its ID
        ids = requests.get(url_season).json()
        for date in ids['dates']:
            for game in date['games']:

                # get the json file of the game from the api
                game_url = game['link']
                game_info = requests.get('https://statsapi.web.nhl.com' + game_url).json()

                # save file
                f_path_save = os.path.join(directory, str(game['gamePk']) + '.json')
                with open(f_path_save, 'w') as file:
                    json.dump(game_info, file)

                games_info.append(game_info)

        print('All the games are saved at', directory)
        return games_info

    def add_season(self, season):
        """

        Add the games from a new season to self.all_games

        Args:
            season: int or string of the name of the season to add, ex: 2017 for the 20172018 season

        """
        if season in self.season:
            print('season {} already loaded')
        else:
            self.season.append(int(season))

            # either load or download the season's games
            if os.path.isdir(os.path.join(self.filepath, str(season))):
                # if folder is empty, then download, else load
                if len(os.listdir(self.filepath)) == 0:
                    self.all_games[season] = self.__download(season)
                else:
                    self.all_games[season] = self.__load(season)
            else:
                self.all_games[season] = self.__download(season)

    def __add__(self, other):
        """

        Combine two GamesInfo object together. Add all the games from one GamesInfo to the other.
        Args:
            other: GamesInfo object, the one we want to combine with this object.
        Returns: new GamesInfo with all the games from both self and other
        """
        seasons = self.season
        filepath = self.filepath
        all_games = self.all_games

        # add season and the games from it to the games from self
        for season in other.all_games:
            if season not in seasons:
                seasons.append(season)
                all_games[season] = other.all_games[season]

        return GamesInfo(seasons, filepath, all_games)

    def get_regular_games(self):
        """

        Get only the regular games data from all_games

        Returns: dictionary of the regular games where keys are the season and values are lists of games data

        """
        regular_game = dict()
        for s in self.season:
            regular_game[s] = []
            for game in self.all_games[s]:
                if game['gameData']['game']['type'] == 'R':
                    regular_game[s].append(game)
        return regular_game

    def get_playoff_games(self):
        """

        Get only the playoff games data from all_games

        Returns: dictionary of the playoff games where keys are the season and values are lists of games data

        """
        playoff_game = dict()
        for s in self.season:
            playoff_game[s] = []
            for game in self.all_games[s]:
                if game['gameData']['game']['type'] == 'P':
                    playoff_game[s].append(game)
        return playoff_game
