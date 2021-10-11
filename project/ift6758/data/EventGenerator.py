from typing import Dict, List
from numpy import show_config
import pandas as pd


class EventGenerator:
    """
        Class that builds a list of desired events from liveData and ultimately generates a dataframe with the 
        selected features. This class points to TidyEvent class to generate a dictionnary to be able to create the dataframe.

        Returns: Dataframe
    """

    def __init__(self, game_pk, live_events, target_events=['SHOT', 'GOAL']) -> None:
        self.game_pk = game_pk
        self.live_events = live_events

        self.target_events = target_events

        self.events = []
        self.events_df = None
        self.event_types = {}

    def build(self) -> List:
        """
            Method that generates a list of features that will be used to generate the final dataframe.
            It will only select features that originate from the target_events argument in the __init__ method (In this case, SHOT and GOAL)

            Returns: list
        """

        for event in self.live_events:
            # Generate all types of events in our game
            event_type = self._fold_event_types(event)

            if event_type in self.target_events:
                # Build tidy event object
                tidy_event = TidyEvent(
                    self.game_pk,
                    event['about']['eventIdx'], event_type,
                    event['team']['id'],
                    event['about']['period'], event['about']['periodType'], event['about']['periodTime'],
                    event['about']['dateTime']
                )

                # Only consider empty net, when goals are not scored in a shootout
                if event_type == 'GOAL' and event['about']['periodType'] != 'SHOOTOUT':
                    tidy_event.set_empty_net(event['result']['emptyNet'])

                # Setup players involved in the event
                for player in event['players']:
                    if player['playerType'] == 'Shooter':
                        tidy_event.set_player_shooter(player['player']['fullName'])

                    elif player['playerType'] == 'Scorer':
                        tidy_event.set_player_scorer(player['player']['fullName'])
                        tidy_event.set_player_shooter(player['player']['fullName'])

                    elif player['playerType'] == 'Goalie':
                        tidy_event.set_player_goalie(player['player']['fullName'])
                # Analyse strength type of GOAL scored
                if event_type == 'GOAL':
                    tidy_event.set_goal_strength(event['result']['strength']['name'])
                
                # Addind secondaryType, x and y coordinates as conditional since some games seem to be missing this key
                if 'secondaryType' in event['result'].keys():
                    tidy_event.set_shot_type(event['result']['secondaryType'])
                if 'x' in event['coordinates'].keys():
                    tidy_event.set_x_coordinate(event['coordinates']['x'])
                if 'y' in event['coordinates'].keys():
                    tidy_event.set_y_coordinate(event['coordinates']['y'])

                    # We can add assisting players here in the future


                self.events.append(tidy_event)

        return self.convert_to_dataframe()


    def convert_to_dataframe(self) -> pd.DataFrame:
        """
            Method converting a dictionnary into a dataframe

            Returns: Dataframe
        """
        self.events_df = pd.DataFrame.from_records([event.to_dict() for event in self.events])


        return self.events_df

    def _fold_event_types(self, event) -> str:
        """
            Method to replace the EventTypeId event type with a desccription if it is missing from source.

            Returns: New eventTypeId generated from desription event type
        """
        if event['result']['eventTypeId'] not in self.event_types:
            self.event_types[event['result']['eventTypeId']] = event['result']['description']

        return event['result']['eventTypeId']



class TidyEvent:
    """
        Class that generates a dictionnary from selected features in liveData.

        Returns: Dictionnary
    """

    def __init__(self, game_pk, event_index, event_type, team_id, period, period_type, period_time, datetime, coordinate_x= None, coordinate_y=None, goal_strength=None, shot_type=None,player_shooter=None, player_scorer=None, player_goalie=None, empty_net=None) -> None:
        self.game_pk = game_pk
        self.event_index = event_index
        self.event_type = event_type
        self.shot_type = shot_type
        self.team_id = team_id
        self.period = period
        self.period_type = period_type
        self.period_time = period_time
        self.datetime = datetime
        self.coordinate_x = coordinate_x
        self.coordinate_y = coordinate_y
        self.player_shooter = player_shooter
        self.player_scorer = player_scorer
        self.player_goalie = player_goalie
        self.empty_net = empty_net
        self.goal_strength  = goal_strength

    def set_empty_net(self, empty_net: bool) -> None:
        self.empty_net = empty_net

    def set_player_goalie(self, player_goalie) -> None:
        self.player_goalie = player_goalie

    def set_player_shooter(self, player_shooter) -> None:
        self.player_shooter = player_shooter

    def set_shot_type(self, shot_type) -> None:
        self.shot_type = shot_type

    def set_player_scorer(self, player_scorer) -> None:
        self.player_scorer = player_scorer
    
    def set_goal_strength(self, goal_strength) -> None:
        self.goal_strength = goal_strength

    def set_x_coordinate(self, coordinate_x) -> None:
        self.coordinate_x = coordinate_x
    
    def set_y_coordinate(self, coordinate_y) -> None:
        self.coordinate_y = coordinate_y

    def to_dict(self) -> Dict:
        """
            Method that generates dicitonnary from selected features.

            Returns: Dictionnary
        """
        return {
            'game_pk': self.game_pk,
            'event_index': self.event_index,
            'event_type': self.event_type,
            'shot_type': self.shot_type,
            'goal_strength' : self.goal_strength,
            'team_id': self.team_id,
            'period': self.period,
            'period_type': self.period_type,
            'period_time': self.period_time,
            'datetime': self.datetime,
            'coordinate_x': self.coordinate_x,
            'coordinate_y': self.coordinate_y,
            'player_shooter': self.player_shooter,
            'player_scorer': self.player_scorer,
            'player_goalie': self.player_goalie,
            'empty_net': self.empty_net,
        }