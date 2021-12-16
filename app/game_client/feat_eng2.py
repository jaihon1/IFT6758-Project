import pandas as pd
import os
import datetime as dt
from datetime import timedelta
import numpy as np
from pathlib import Path


def add_new_features(dataframe):
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
    return df_filtered

