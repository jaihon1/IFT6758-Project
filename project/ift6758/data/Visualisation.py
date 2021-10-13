import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os


dirpath = os.path.join(os.path.dirname(__file__))
file_path = os.path.join(dirpath, 'games_data_all_seasons.csv')
df = pd.read_csv(file_path)

df['game_pk'] = df['game_pk'].astype(str)
df_2016= df[df['game_pk'].str.startswith('2016')]
df_2016['game_pk'] = df_2016['game_pk'].astype(int)
df_2016_goals = df_2016[df_2016['event_type'] == 'GOAL']
df_2016_shots = df_2016[df_2016['event_type'] == 'SHOT']
Number_of_shots_per_Shot_types = df_2016_shots['shot_type'].value_counts()
Number_of_Goals_per_Shot_types = df_2016_goals['shot_type'].value_counts()


new_index= ['Wrist Shot', 'Slap Shot', 'Snap Shot', 'Backhand', 'Tip-In', 'Deflected', 'Wrap-around']
Number_of_shots = Number_of_shots_per_Shot_types.reindex(new_index)
Number_of_Goals = Number_of_Goals_per_Shot_types.reindex(new_index)


#### Overlap of the SHOTS and GOALS : BARPLOT
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax2 = ax.twinx()
ax.set_title('Shots and Goals in function of shot type over all teams in the 2016 season')
ax.set_xlabel('Types of Shots', fontsize=12)
sns.barplot(Number_of_shots.index, Number_of_shots_per_Shot_types.values, alpha=0.4)
sns.barplot(Number_of_Goals.index, Number_of_Goals_per_Shot_types.values, alpha=0.4)
ax2.set_ylim(0, 25000)
ax.set_ylim(0, 25000)
ax.set_ylabel('Absolute frequency')
ax2.get_yaxis().set_visible(False)
plt.show()

### Most dangerous type of Shots
Dangerous_Shots = Number_of_Goals / Number_of_shots * 100
sns.barplot(Dangerous_Shots.index, Dangerous_Shots.values, alpha=0.5)
plt.title('Most dangerous type of shots in the 2016 season')
plt.ylabel('Percentage of shots that resulted in goals', fontsize=12)
plt.xlabel('Types of Shots', fontsize=12)
plt.xticks(fontsize=10, rotation=45)
plt.show()
