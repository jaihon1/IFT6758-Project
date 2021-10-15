import matplotlib.pyplot as plt
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
Number_of_shots_per_Shot_types = df_2016_shots['shot_type'].value_counts()
Number_of_Goals_per_Shot_types = df_2016_goals['shot_type'].value_counts()

### New Index to make a better and more symetric graph
new_index= ['Wrist Shot', 'Slap Shot', 'Snap Shot', 'Backhand', 'Tip-In', 'Deflected', 'Wrap-around']
Number_of_shots = Number_of_shots_per_Shot_types.reindex(new_index)
Number_of_Goals = Number_of_Goals_per_Shot_types.reindex(new_index)


#### Overlap of the SHOTS and GOALS : BARPLOT
fig, ax = plt.subplots(figsize=(12, 6))
ax2 = ax.twinx()
ax.set_title('Shots and Goals per number of shot types over all teams in the 2016 season')
ax.set_xlabel('Types of Shots', fontsize=12)
sns.barplot(Number_of_shots.index, Number_of_shots_per_Shot_types.values, alpha=0.4)
sns.barplot(Number_of_Goals.index, Number_of_Goals_per_Shot_types.values, alpha=0.4)
ax2.set_ylim(0, 25000)
ax.set_ylim(0, 25000)
ax.set_ylabel('Absolute frequency')
ax2.get_yaxis().set_visible(False)
plt.show()

### Most dangerous type of Shots
Dangerous_Shots = Number_of_Goals / (Number_of_shots + Number_of_Goals) * 100
sns.barplot(Dangerous_Shots.index, Dangerous_Shots.values, alpha=0.5)
plt.title('Most dangerous type of shots in the 2016 season')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Types of Shots', fontsize=12)
plt.xticks(fontsize=10, rotation=45)
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

### define the bins : 20 bins from  minimum value to maximum value
bins = np.linspace(min(events_new_coor_2018['shot_distance']), max(events_new_coor_2018['shot_distance']), 21)
# add labels if desired
labels = ["1-5.8", "5.8-10.7", "10.7-15.5", "15.5-20.4", "20.4-25.2", "25.2-30.1", "30.1-34.9", "34.9-39.8", "39.8-44.6", "44.6-49.5", "49.5-54.3", "54.3-59.2", "59.2-64.0", "64.0-68.9", "68.9-73.7", "73.7-78.6", "78.6-83.4", "83.4-88.3", "88.3-93.1", "93.1-98.0"]

### add the bins to the dataframe :
events_new_coor_2018["bin_Shot_distance"] = pd.cut(events_new_coor_2018["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage = []
for i in range(len(labels)):
    temp_df = events_new_coor_2018[events_new_coor_2018["bin_Shot_distance"]==labels[i]]
    list_of_Percentage.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))

### Plotting the BarPlot
sns.barplot(labels, list_of_Percentage, alpha=0.4)
plt.xticks(fontsize=9, rotation=45)
plt.ylim(0, 25)
plt.title('Chance of scoring a goal in function of shot distance: Season 2018')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Shot distance', fontsize=12)
plt.show()

### Season 2019
### Generating Dataframe For the Season (2019)

events_new_coor_2019 = events_new_coor[events_new_coor['game_pk'].str.startswith('2019')]
q1 = 89  # valeur absolue (location sur l'axe X)
q2 = 0  # valeur absolue (location sur l'axe Y)
events_new_coor_2019['shot_distance'] = [np.sqrt((q1 - abs(p1))** 2 + (q2 - abs(p2))** 2) for p1, p2, in zip(events_new_coor_2019['coordinate_x'], events_new_coor_2019['coordinate_y'])]

### define the bins : 20 bins from  minimum value to maximum value
bins = np.linspace(min(events_new_coor_2019['shot_distance']), max(events_new_coor_2019['shot_distance']), 21)
# add labels if desired
labels = ["1-5.8", "5.8-10.7", "10.7-15.5", "15.5-20.4", "20.4-25.2", "25.2-30.1", "30.1-34.9", "34.9-39.8", "39.8-44.6", "44.6-49.5", "49.5-54.3", "54.3-59.2", "59.2-64.0", "64.0-68.9", "68.9-73.7", "73.7-78.6", "78.6-83.4", "83.4-88.3", "88.3-93.1", "93.1-98.0"]

### add the bins to the dataframe :
events_new_coor_2019["bin_Shot_distance"] = pd.cut(events_new_coor_2019["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage = []
for i in range(len(labels)):
    temp_df = events_new_coor_2019[events_new_coor_2019["bin_Shot_distance"]==labels[i]]
    list_of_Percentage.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))

### Plotting the BarPlot
sns.barplot(labels, list_of_Percentage, alpha=0.4)
plt.xticks(fontsize=9, rotation=45)
plt.ylim(0, 25)
plt.title('Chance of scoring a goal in function of shot distance: Season 2019')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Shot distance', fontsize=12)
plt.show()
print(list_of_Percentage)
### Season 2020
### Generating Dataframe For the Season (2020)

events_new_coor_2020 = events_new_coor[events_new_coor['game_pk'].str.startswith('2020')]
q1 = 89  # valeur absolue (location sur l'axe X)
q2 = 0  # valeur absolue (location sur l'axe Y)
events_new_coor_2020['shot_distance'] = [np.sqrt((q1 - abs(p1))** 2 + (q2 - abs(p2))** 2) for p1, p2, in zip(events_new_coor_2020['coordinate_x'], events_new_coor_2020['coordinate_y'])]

### define the bins : 20 bins from  minimum value to maximum value
bins = np.linspace(min(events_new_coor_2020['shot_distance']), max(events_new_coor_2020['shot_distance']), 21)
# add labels if desired
labels = ["1-5.8", "5.8-10.7", "10.7-15.5", "15.5-20.4", "20.4-25.2", "25.2-30.1", "30.1-34.9", "34.9-39.8", "39.8-44.6", "44.6-49.5", "49.5-54.3", "54.3-59.2", "59.2-64.0", "64.0-68.9", "68.9-73.7", "73.7-78.6", "78.6-83.4", "83.4-88.3", "88.3-93.1", "93.1-98.0"]

### add the bins to the dataframe :
events_new_coor_2020["bin_Shot_distance"] = pd.cut(events_new_coor_2020["shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
list_of_Percentage = []
for i in range(len(labels)):
    temp_df = events_new_coor_2020[events_new_coor_2020["bin_Shot_distance"]==labels[i]]
    list_of_Percentage.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))
print(list_of_Percentage)
### Plotting the BarPlot
sns.barplot(labels, list_of_Percentage, alpha=0.4)
plt.xticks(fontsize=9, rotation=45)
plt.ylim(0, 25)
plt.title('Chance of scoring a goal in function of shot distance: Season 2020')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Shot distance', fontsize=12)
plt.show()


### Goal Percentage(# Goals/ # total de shots)
### adding the column  to the dataframe :
for events_new_coor_2020["bin_Shot_distance"], temp_df in events_new_coor_2020.groupby('shot_type'):
    list_of_Goals_Percentage = []
    for i in range(len(labels)):
        if temp_df == events_new_coor_2020[events_new_coor_2020["bin_Shot_distance"]==labels[i]]:
            list_of_Goals_Percentage.append(round(temp_df["event_type"].value_counts()[1] / (temp_df["event_type"].value_counts()[0] + temp_df["event_type"].value_counts()[1]) * 100, 2))


### Plotting the BarPlot
sns.barplot(events_new_coor_2020['shot_type'], list_of_Goals_Percentage, alpha=0.4)
plt.xticks(fontsize=9, rotation=45)
plt.ylim(0, 25)
plt.title('Chance of scoring a goal in function of shot distance: Season 2020')
plt.ylabel('Percentage of scoring shots', fontsize=12)
plt.xlabel('Shot distance', fontsize=12)
plt.show()