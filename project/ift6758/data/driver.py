from GamesInfo import GamesInfo
from EventGenerator import EventGenerator
import json
import os




def main():
    # seasons = [2019]
    # games_info = GamesInfo(seasons)


    data = None

    dirpath = os.path.join(os.path.dirname(__file__))
    filepath = dirpath + '/games_data/2019/2019020058.json'

    with open(filepath) as file:
        data = json.load(file)


    live_events = data['liveData']['plays']['allPlays']
    game_pk = data['gamePk']
    game = EventGenerator(game_pk, live_events)
    game.build()


if __name__ == "__main__":
    main()