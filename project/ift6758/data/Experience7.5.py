import os
import pandas as pd
from comet_ml import Experiment

#print(os.environ["KEY"])
experiment = Experiment(api_key=os.environ.get('COMET_API_KEY'), project_name="ift6758-project", workspace="jaihon")
subset_df = pd.read_csv(r"C:\Users\samib\Desktop\ProjectIFT6758\IFT6758-Project\project\ift6758\data\games_data\games_data_all_seasons.csv", encoding='UTF-8')
subset_df_WPG_WSH = subset_df[(subset_df['team_id'] == 'WSH') | (subset_df['team_id'] == 'WPG')]
# Filter by March 12, 2018
subset_df_WPG_WSH = subset_df_WPG_WSH[subset_df_WPG_WSH['datetime'].str.contains('2018-03-12')]

experiment.log_dataframe_profile(subset_df_WPG_WSH, name='wpg_v_wsh_2017021065', dataframe_format='csv')
# #exp.log_metrics({"auc": auc, “acc”: acc, “loss”:Expercd  loss})as usual