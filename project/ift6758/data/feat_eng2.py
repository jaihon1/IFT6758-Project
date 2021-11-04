
import pandas as pd
import os
import datetime as dt
from datetime import timedelta
import numpy as np

csv_path = os.path.join(os.path.dirname(__file__),'games_data/games_data_all_seasons.csv')
df = pd.read_csv(csv_path)
df_filtered = df.loc[:,['period', 'period_time','event_type', 'shot_type', 'coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'empty_net']]
df_filtered['game_seconds'] = [dt.datetime.strptime(i, '%M:%S').second + dt.datetime.strptime(i, '%M:%S').minute*60 for i in df_filtered['period_time']]
df_filtered.drop(columns='period_time', inplace=True)
df_additional_info = df_filtered.copy()
df_additional_info['Last_event_type'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('event_type')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['X_Coordinate_of_last_event'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_x')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Y_Coordinate_of_last_event'] = [df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_y')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Time_from_last_event'] = [df_additional_info.iloc[i, df_additional_info.columns.get_loc('game_seconds')] - df_additional_info.iloc[i-1, df_additional_info.columns.get_loc('game_seconds')] if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_additional_info['Distance_from_last_event'] = [np.sqrt((df_additional_info.iloc[i,df_additional_info.columns.get_loc('coordinate_x')] - df_additional_info.iloc[i-1, df_additional_info.columns.get_loc('coordinate_x')])**2+ (df_additional_info.iloc[i, df_additional_info.columns.get_loc('coordinate_y')] - df_additional_info.iloc[i-1,df_additional_info.columns.get_loc('coordinate_y')])**2)if df_additional_info.iloc[i,df_additional_info.columns.get_loc('event_type')] == 'SHOT' else np.nan for i in range(len(df_additional_info))]
df_shots_filtered_info = df_additional_info[df_additional_info['event_type'] == "SHOT"]
print(df_additional_info)
print(df_shots_filtered_info)
