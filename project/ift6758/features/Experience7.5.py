import os
import pandas as pd
from comet_ml import Experiment

experiment = Experiment(api_key=os.environ.get('COMET_API_KEY'), project_name="ift6758-project", workspace="jaihon")
file_path = os.path.join(os.path.dirname(__file__), 'games_data\games_data_all_season.csv')
subset_df = pd.read_csv(file_path, encoding='UTF-8')
subset_df_WPG_WSH = subset_df[(subset_df['team_id'] == 'WSH') | (subset_df['team_id'] == 'WPG')]
# Filter by March 12, 2018
subset_df_WPG_WSH = subset_df_WPG_WSH[subset_df_WPG_WSH['datetime'].str.contains('2018-03-12')]

experiment.log_dataframe_profile(subset_df_WPG_WSH, name='wpg_v_wsh_2017021065', dataframe_format='csv')
