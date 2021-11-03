import math
from typing import Dict, List
import pandas as pd


class EventGenerator:
    """
        Class that builds a list of desired events from liveData and ultimately generates a dataframe with the 
        selected features. This class points to TidyEvent class to generate a dictionnary to be able to create the dataframe.
    """

    def __init__(self, game_pk, home, away, sides, live_events, target_events=['SHOT', 'GOAL']) -> None:
        self.game_pk = game_pk
        self.live_events = live_events

        self.target_events = target_events

        self.events = []
        self.events_df = None
        self.event_types = {}

        self.home = home
        self.away = away
        self.sides = sides

    def build(self) -> pd.DataFrame:
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
                    event['team']['triCode'],
                    event['about']['period'], event['about']['periodType'], event['about']['periodTime'],
                    event['about']['dateTime'],
                )

                # add where the team's goal is
                if event['about']['periodType'] != 'SHOOTOUT':
                    tidy_event.set_team_side(self.sides[event['about']['period']][event['team']['triCode']])

                # Only consider empty net, when goals are not scored in a shootout
                tidy_event.set_empty_net(0)
                if event_type == 'GOAL' and event['about']['periodType'] != 'SHOOTOUT':
                    tidy_event.set_empty_net(int(event['result']['emptyNet']))

                if event_type == 'GOAL':
                    tidy_event.set_is_goal(1)
                else:
                    tidy_event.set_is_goal(0)

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
                if 'x' in event['coordinates'].keys() and 'y' in event['coordinates'].keys():
                    if tidy_event.team_side is not None:
                        team_side = tidy_event.team_side
                        coordinate_x = tidy_event.coordinate_x
                        if team_side == 'right' and tidy_event.coordinate_x < 0:
                            coordinate_x = abs(coordinate_x)
                    else:
                        coordinate_x = abs(tidy_event.coordinate_x)
                    net_coordinate = (89, 0)
                    distance_x = net_coordinate[0] - coordinate_x
                    distance_y = tidy_event.coordinate_y - net_coordinate[1]
                    tidy_event.set_distance_net(math.sqrt(distance_x ** 2 + distance_y ** 2))
                    angle = math.degrees(math.atan2(distance_y, distance_x))
                    tidy_event.set_angle_net(angle)

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
    """

    def __init__(self, game_pk, event_index, event_type, team_id, period, period_type, period_time, datetime,
                 coordinate_x=None, coordinate_y=None, goal_strength=None, shot_type=None, player_shooter=None,
                 player_scorer=None, player_goalie=None, empty_net=None, is_goal=None, team_side=None,
                 distance_net=None, angle_net=None) -> None:
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
        self.is_goal = is_goal
        self.goal_strength = goal_strength
        self.team_side = team_side
        self.distance_net = distance_net
        self.angle_net = angle_net

    def set_empty_net(self, empty_net: int) -> None:
        self.empty_net = empty_net

    def set_is_goal(self, is_goal) -> None:
        self.is_goal = is_goal

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

    def set_team_side(self, side) -> None:
        self.team_side = side

    def set_distance_net(self, distance_net) -> None:
        self.distance_net = distance_net

    def set_angle_net(self, angle_net) -> None:
        self.angle_net = angle_net

    def to_dict(self) -> Dict:
        """
            Method that generates dictionnary from selected features.

            Returns: Dictionnary
        """
        return {
            'game_pk': self.game_pk,
            'event_index': self.event_index,
            'event_type': self.event_type,
            'shot_type': self.shot_type,
            'goal_strength': self.goal_strength,
            'team_id': self.team_id,
            'team_side': self.team_side,
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
            'is_goal': self.is_goal,
            'distance_net': self.distance_net,
            'angle_net': self.angle_net
        }
