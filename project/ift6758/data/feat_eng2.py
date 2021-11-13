import pandas as pd
import os
import datetime as dt
from datetime import timedelta
import numpy as np
from pathlib import Path
csv_path = os.path.join(os.path.dirname(__file__),'C:\Users\samib\Desktop\ProjectIFT6758\IFT6758-Project\project\ift6758\data\games_data_all_seasons.csv')
df = pd.read_csv(csv_path)
df_filtered = df.loc[:, ['game_pk','period', 'period_type', 'period_time','event_type', 'shot_type', 'coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'empty_net', 'previous_event_type', 'previous_event_x_coord', 'previous_event_y_coord', 'previous_event_period', 'previous_event_period_time']]
df_filtered['game_seconds'] = [dt.datetime.strptime(i, '%M:%S').second + dt.datetime.strptime(i, '%M:%S').minute*60 for i in df_filtered['period_time']]
df_filtered['previous_event_seconds'] = [dt.datetime.strptime(i, '%M:%S').second + dt.datetime.strptime(i, '%M:%S').minute*60 for i in df_filtered['previous_event_period_time']]
df_filtered['max_period'] = df_filtered['period'].groupby(df_filtered['game_pk']).transform('max')
df_filtered['game_seconds'] = [i + (period-1)*20*60 if max == 3 else i + (3*20*60 + (period -1 - 3)*5*60) if max >3 else 'Error' for i, period, max in zip(df_filtered['game_seconds'],
    df_filtered['period'], df_filtered['max_period'])]
df_filtered['previous_event_seconds'] = [i + (period-1)*20*60 if max == 3 else i + (3*20*60 + (period -1 - 3)*5*60) if max >3 else 'Error' for i, period, max in zip(df_filtered['previous_event_seconds'],
    df_filtered['period'], df_filtered['max_period'])]
df_filtered.drop(columns=['period_time','previous_event_period_time', 'previous_event_period', 'max_period'], inplace=True)
df_filtered['shot_last_event_delta'] = [t1 - t0 for t1, t0 in zip(df_filtered['game_seconds'], df_filtered['previous_event_seconds'])]
df_filtered['shot_last_event_distance'] = [np.sqrt((x2-x1)**2 - (y2 - y1)**2) for x1, x2, y1, y2 in zip(df_filtered['previous_event_x_coord'], df_filtered['coordinate_x'], df_filtered['previous_event_y_coord'], df_filtered['coordinate_y'])]
print(df_filtered[df_filtered['coordinate_y'] == np.nan])
### Adding the Rebound Feature
df_filtered.loc[df_filtered['previous_event_type'] == 'SHOT', 'Rebound'] = 'True'
df_filtered.loc[df_filtered['previous_event_type'] != 'SHOT', 'Rebound'] = 'False'
# ### Adding Change in shot angle feature
myradians = np.arctan2(df_filtered['previous_event_y_coord']-df_filtered['coordinate_y'], df_filtered['previous_event_x_coord']-df_filtered['coordinate_x'])
df_filtered.loc[df_filtered['Rebound'] == 'True', 'Change_in_shot_angle'] = np.degrees(myradians)
df_filtered.loc[df_filtered['Rebound'] == 'False', 'Change_in_shot_angle'] = '0'
# ### Adding Speed feature: distance from the previous event, divided by the time since the previous event
df_filtered['Speed'] = df_filtered['shot_last_event_distance'] / df_filtered['shot_last_event_delta']
df_filtered.info()
print(df_filtered.iloc[:10,:])

# ### Bonus
# ### Time since the power-play started ( a situation in which one team has more players than the other because that team has had one or more players temporarily sent off;
# # temporary numerical advantage because an opposing player or players are in the penalty box.)
# # 3 types of Penalities
# # a minor (2 minutes): the least severe: down a skater for 5 on 4
#
# # double minor (4 minutes): down a skater for 5 on 4: 2 minor penalties combined in one
# # if a Goal while double minor penalty, one of teh minor penalty will expire
# # a major, such as a 5-minute misconduct: down a skater for 5 minutes: the penalty is never wiped even if a goal is scored by the opposing team.
# Minor = 2
# df_additional_info['Time since the power-play started'] = df_additional_info['Time_from_last_event']
#
#
# ### Number of friendly non-goalie skaters on the ice
# #5 on 4 5 on 3 4 on 3
# ### Number of opposing non-goalie skaters on the ice