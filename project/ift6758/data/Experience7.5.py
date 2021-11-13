import os
import pandas as pd
from comet_ml import Experiment

#print(os.environ["KEY"])
experiment = Experiment(api_key=os.environ.get("KEY"),  project_name="milestone_2")
subset_df = pd.read_csv(r"C:\Users\samib\Desktop\ProjectIFT6758\IFT6758-Project\project\ift6758\data\games_data\games_data_all_seasons.csv", parse_dates=['game_pk'], date_parse="2018020007", encoding='UTF-8')
experiment.log_dataframe_profile(subset_df, name='wpg_v_wsh_2017021065', dataframe_format='csv')
#exp.log_metrics({"auc": auc, “acc”: acc, “loss”: loss})as usual