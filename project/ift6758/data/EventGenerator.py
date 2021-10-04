from typing import Dict, List
import pandas as pd


class EventGenerator:
    """
        ...
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
            ...
        """

        for event in self.live_events:
            # Generate all types of events in our game
            event_type = self._fold_event_types(event)

            if event_type in self.target_events:
                # Build tidy event object
                tidy_event = TidyEvent(
                    self.game_pk,
                    event['about']['eventIdx'], event_type, event['result']['secondaryType'],
                    event['team']['id'],
                    event['about']['period'], event['about']['periodType'], event['about']['periodTime'],
                    event['about']['dateTime'],
                    event['coordinates']['x'], event['coordinates']['y']
                )

                # Only consider empty net, when goals are not scored in a shootout
                if event_type == 'GOAL' and event['about']['periodType'] != 'SHOOTOUT':
                    tidy_event.set_empty_net = event['result']['emptyNet']

                # Setup players involved in the event
                for player in event['players']:
                    if player['playerType'] == 'Shooter':
                        tidy_event.set_player_shooter(player['player']['fullName'])

                    elif player['playerType'] == 'Scorer':
                        tidy_event.set_player_scorer(player['player']['fullName'])
                        tidy_event.set_player_shooter(player['player']['fullName'])

                    elif player['playerType'] == 'Goalie':
                        tidy_event.set_player_goalie(player['player']['fullName'])

                    # We can add assisting players here in the future


                self.events.append(tidy_event)

        return self.convert_to_dataframe()


    def convert_to_dataframe(self) -> pd.DataFrame:
        """
            ...
        """
        self.events_df = pd.DataFrame.from_records([event.to_dict() for event in self.events])

        print(self.events_df[20:])

        return self.events_df

    def _fold_event_types(self, event) -> str:
        """
            ...
        """
        if event['result']['eventTypeId'] not in self.event_types:
            self.event_types[event['result']['eventTypeId']] = event['result']['description']

        return event['result']['eventTypeId']



class TidyEvent:
    """
        ...
    """

    def __init__(self, game_pk, event_index, event_type, shot_type, team_id, period, period_type, period_time, datetime, coordinate_x, coordinate_y, player_shooter=None, player_scorer=None, player_goalie=None, empty_net=None) -> None:
        self.game_pk = game_pk
        self.event_inedx = event_index
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

    def set_empty_net(self, empty_net: bool) -> None:
        self.empty_net = empty_net

    def set_player_goalie(self, player_goalie) -> None:
        self.player_goalie = player_goalie

    def set_player_shooter(self, player_shooter) -> None:
        self.player_shooter = player_shooter

    def set_player_scorer(self, player_scorer) -> None:
        self.player_scorer = player_scorer

    def to_dict(self) -> Dict:
        """
            ...
        """
        return {
            'game_pk': self.game_pk,
            'event_index': self.event_inedx,
            'event_type': self.event_type,
            'shot_type': self.shot_type,
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