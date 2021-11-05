import pandas as pd
import os
import datetime as dt
from datetime import timedelta
import numpy as np
from pathlib import Path
#csv_path = os.path.join(os.path.dirname(__file__),'g')
path = Path("/Users/samib/Desktop/ProjectIFT6758/IFT6758-Project/project/ift6758/data/games_data/games_data_all_seasons.csv")
df = pd.read_csv(path)
df_filtered = df.loc[:, ['period', 'period_time','event_type', 'shot_type', 'coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'empty_net']]
df_filtered['game_seconds'] = [dt.datetime.strptime(i, '%M:%S').second + dt.datetime.strptime(i, '%M:%S').minute*60 for i in df_filtered['period_time']]
df_filtered.drop(columns='period_time', inplace=True)
df_additional_info = df_filtered.copy()
df_additional_info['Last_event_type'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('event_type')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['X_Coordinate_of_last_event'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_x')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Y_Coordinate_of_last_event'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_y')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Time_from_last_event'] = [df_additional_info.iloc[i, df_additional_info.columns.get_loc('game_seconds')] - df_additional_info.iloc[i-1, df_additional_info.columns.get_loc('game_seconds')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Distance_from_last_event'] = [np.sqrt((df_additional_info.iloc[i,df_additional_info.columns.get_loc('coordinate_x')] - df_additional_info.iloc[i-1, df_additional_info.columns.get_loc('coordinate_x')])**2+ (df_additional_info.iloc[i, df_additional_info.columns.get_loc('coordinate_y')] - df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_y')])**2)if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_shots_filtered_info = df_additional_info[df_additional_info['event_type'] == "SHOT"]
### Adding the Rebound Feature
df_additional_info.loc[df_additional_info['Last_event_type'] == 'SHOT', 'Rebound'] = 'True'
df_additional_info.loc[df_additional_info['Last_event_type'] != 'SHOT', 'Rebound'] = 'False'
print(df_additional_info)
### Adding Change in shot angle feature
myradians = np.arctan2(df_additional_info['Y_Coordinate_of_last_event']-df_additional_info['coordinate_y'], df_additional_info['X_Coordinate_of_last_event']-df_additional_info['coordinate_x'])
df_additional_info.loc[df_additional_info['Rebound'] == 'True', 'Change_in_shot_angle'] = np.degrees(myradians)
df_additional_info.loc[df_additional_info['Rebound'] == 'False', 'Change_in_shot_angle'] = '0'
### Adding Speed feature: distance from the previous event, divided by the time since the previous event
df_additional_info['Speed'] = df_additional_info['Distance_from_last_event'] / df_additional_info['Time_from_last_event']
print(df_additional_info)

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