import matplotlib.pyplot as plt
from nest_asyncio import apply
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import seaborn as sns
import os

### DataFrame
dirpath = os.path.join(os.path.dirname(__file__))
file_path = os.path.join(dirpath, 'games_data_all_seasons.csv')
df = pd.read_csv(file_path)

### Part1

### Number_of_shots_and_number_of_Goals_per_Shot_types

df['game_pk'] = df['game_pk'].astype(str)
df_2016= df[df['game_pk'].str.startswith('2016')]
df_2016['game_pk'] = df_2016['game_pk'].astype(int)
df_2016_goals = df_2016[df_2016['event_type'] == 'GOAL']
df_2016_shots = df_2016[df_2016['event_type'] == 'SHOT']
Number_of_shots_per_Shot_types = df_2016_shots['shot_type'].value_counts() + df_2016_goals['shot_type'].value_counts()
Number_of_Goals_per_Shot_types = df_2016_goals['shot_type'].value_counts()
### New Index to make a better and more symetric graph
new_index= ['Wrist Shot', 'Slap Shot', 'Snap Shot', 'Backhand', 'Tip-In', 'Deflected', 'Wrap-around']
Number_of_shots = Number_of_shots_per_Shot_types.reindex(new_index)
Number_of_Goals = Number_of_Goals_per_Shot_types.reindex(new_index)


#### Overlap of the SHOTS and GOALS : BARPLOT
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
ax2 = ax.twinx()
ax.set_title('Shots and Goals per number of shot types over all teams in the 2016 season')
ax.set_xlabel('Types of Shots', fontsize=12)
sns.barplot(Number_of_shots.index, Number_of_shots.values, alpha=0.4)
sns.barplot(Number_of_Goals.index, Number_of_Goals.values, alpha=0.4)
ax2.set_ylim(0, 25000)
ax.set_ylim(0, 25000)
ax.set_ylabel('Absolute frequency', size=12)
ax2.get_yaxis().set_visible(False)
plt.show()

### Most dangerous type of Shots
Dangerous_Shots = Number_of_Goals / (Number_of_shots + Number_of_Goals) * 100
Dangerous_Shots = Dangerous_Shots.sort_values(ascending=False)
print(Dangerous_Shots)
sns.set_theme(style="whitegrid")
sns.barplot(Dangerous_Shots.index, Dangerous_Shots.values)
plt.title('Most dangerous type of shots in the 2016 season')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Types of Shots', fontsize=12)
plt.xticks(fontsize=10, rotation=20)
plt.show()


### Part 2
#### Reading NewDataFrame done by Laurence's Code generating a team_side column
### New Dataset'name is "events_new_coor"
### DataFrame


#events = pd.read_csv(file_path)
events_new_coor = df.copy()[~(df['coordinate_x'].isnull()) & (~df['coordinate_y'].isnull())]

### Adjusting the negative values of coordinate x depending on the team's side

def adjust_negative_value_x(row):
    if row['team_side'] != 'left' and row['team_side'] != 'right':
        row['coordinate_x'] = abs(row['coordinate_x'])
    elif row['team_side'] == 'right' and row['coordinate_x']<0:
        row['coordinate_x'] = abs(row['coordinate_x'])
    return row

events_new_coor = events_new_coor.apply(lambda row: adjust_negative_value_x(row), axis=1)

### Season 2018
### Generating Dataframe For the Season (2018)

events_new_coor_2018 = events_new_coor[events_new_coor['game_pk'].str.startswith('2018')]
q1 = 89  # valeur absolue (location sur l'axe X)
q2 = 0  # valeur absolue (location sur l'axe Y)
events_new_coor_2018['shot_distance'] = [np.sqrt((q1 - abs(p1))** 2 + (q2 - abs(p2))** 2) for p1, p2, in zip(events_new_coor_2018['coordinate_x'], events_new_coor_2018['coordinate_y'])]

### define the bins : 14 bins from  minimum value to maximum value
bins = np.linspace(min(events_new_coor_2018['shot_distance']), max(events_new_coor_2018['shot_distance']), 14)

# add labels if desired
labels = ["1-8.5", "8.5-15.9", "15.9-23.4", "23.4-30.8", "30.8-38.3", "38.3-45.8", "45.8-53.2", "53.2-60.7", "60.7-68.1", "68.1-75.6", "75.6-83.1", "83.1-90.5", "90.5-98.0"]

### add the bins to the dataframe :
events_new_coor_2018["bin_Shot_distance"] = pd.cut(events_new_coor_2018["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage_2018 = []
for i in range(len(labels)):
    temp_df = events_new_coor_2018[events_new_coor_2018["bin_Shot_distance"]==labels[i]]
    list_of_Percentage_2018.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))

### Season 2019
### Generating Dataframe For the Season (2019)

events_new_coor_2019 = events_new_coor[events_new_coor['game_pk'].str.startswith('2019')]
q1 = 89  # valeur absolue (location sur l'axe X)
q2 = 0  # valeur absolue (location sur l'axe Y)
events_new_coor_2019['shot_distance'] = [np.sqrt((q1 - abs(p1))** 2 + (q2 - abs(p2))** 2) for p1, p2, in zip(events_new_coor_2019['coordinate_x'], events_new_coor_2019['coordinate_y'])]

### add the bins to the dataframe :
events_new_coor_2019["bin_Shot_distance"] = pd.cut(events_new_coor_2019["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage_2019 = []
for i in range(len(labels)):
    temp_df = events_new_coor_2019[events_new_coor_2019["bin_Shot_distance"]==labels[i]]
    list_of_Percentage_2019.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))

### Season 2020
### Generating Dataframe For the Season (2020)

events_new_coor_2020 = events_new_coor[events_new_coor['game_pk'].str.startswith('2020')]
q1 = 89  # valeur absolue (location sur l'axe X)
q2 = 0  # valeur absolue (location sur l'axe Y)
events_new_coor_2020['shot_distance'] = [np.sqrt((q1 - abs(p1))** 2 + (q2 - abs(p2))** 2) for p1, p2, in zip(events_new_coor_2020['coordinate_x'], events_new_coor_2020['coordinate_y'])]


### add the bins to the dataframe :
events_new_coor_2020["bin_Shot_distance"] = pd.cut(events_new_coor_2020["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage_2020 = []
for i in range(len(labels)):
    temp_df = events_new_coor_2020[events_new_coor_2020["bin_Shot_distance"]==labels[i]]
    list_of_Percentage_2020.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))

### Plotting the BarPlot
fig, axs = plt.subplots(3)
fig.suptitle('Chance of scoring a goal in function of shot distance for seasons 2018, 2019, 2020')
sns.set_theme(style="whitegrid")
sns.barplot(labels, list_of_Percentage_2018, color='black', alpha=0.6, ax=axs[0])
sns.barplot(labels, list_of_Percentage_2019, color='black', alpha=0.6, ax=axs[1])
sns.barplot(labels, list_of_Percentage_2020, color='black', alpha=0.6, ax=axs[2])
axs[0].set(title='Season 2018')
axs[1].set(title='Season 2019')
axs[2].set(title='Season 2020')
axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)
plt.xticks(fontsize=9, rotation=45)
plt.ylim(0, 25)
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Shot distance (feet)', fontsize=12)
plt.show()


### Goal Percentage(# Goals/ # total de shots) Season 2020

events_new_coor_2020_grouped= events_new_coor_2020.groupby(['shot_type', 'bin_Shot_distance']).apply(lambda x: sum(x['event_type'] == 'GOAL') / (sum(x['event_type'] == 'GOAL')+ sum(x['event_type'] == 'SHOT'))*100)
events_new_coor_2020_grouped_df = pd.DataFrame(events_new_coor_2020_grouped, columns=['shot_efficiency'])
events_new_coor_2020_grouped_df.reset_index(inplace=True)


### Plotting the BarPlot   Season 2020
sns.set_theme(style="whitegrid")
f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})
sns.barplot(x="bin_Shot_distance", y="shot_efficiency", hue="shot_type",data=events_new_coor_2020_grouped_df, ax=ax_top)
sns.barplot(x="bin_Shot_distance", y="shot_efficiency", hue="shot_type",data=events_new_coor_2020_grouped_df, ax=ax_bottom)
ax_top.set_ylim(bottom=80, top=100) 
ax_bottom.set_ylim(0,35)
sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)
ax = ax_top
d = .015 
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax2 = ax_bottom
ax2.set(ylabel=None)
ax_top.get_xaxis().set_visible(False)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.tick_params(axis='x', labelrotation=45 )
ax_bottom.legend_.remove()
ax.set(ylabel='Shot efficiency (Shot to goal)', title='Shot efficiency in function of shot type and shot distance for 2020')
ax_bottom.set(xlabel='Shot distance (feet)')
plt.show()
