import os
import pandas as pd
from comet_ml import Experiment

#print(os.environ["KEY"])
experiment = Experiment(api_key=os.environ.get("KEY"),  project_name="milestone_2")
subset_df = pd.read_csv(r"C:\Users\samib\Desktop\ProjectIFT6758\IFT6758-Project\project\ift6758\data\games_data\games_data_all_seasons.csv", date_parse="2018020007", encoding='UTF-8')
# convert date column into date format
subset_df['game_pk'] = pd.to_datetime(df['game_pk'])

# filter rows on the basis of date
newdf = (subset_df['game_pk'] == '02-02-2018')

# locate rows and access them using .loc() function
newdf = df.loc[newdf]

# print dataframe
print(newdf)
experiment.log_dataframe_profile(new_df, name='wpg_v_wsh_2017021065', dataframe_format='csv')
#exp.log_metrics({"auc": auc, “acc”: acc, “loss”: loss})as usual