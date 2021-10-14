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
Dangerous_Shots = Number_of_Goals / Number_of_shots * 100
sns.barplot(Dangerous_Shots.index, Dangerous_Shots.values, alpha=0.5)
plt.title('Most dangerous type of shots in the 2016')
plt.ylabel('Percentage of Goals on Shots', fontsize=12)
plt.xlabel('Types of Shots', fontsize=12)
plt.xticks(fontsize=10, rotation=45)
plt.show()


### Part 2
#### Reading NewDataFrame done by Laurence's Code generating a team_side column
### New Dataset'name is "events_new_coor"

df = pd.read_csv("/content/games_data_all_seasons.csv")
events = pd.read_csv('/content/games_data_all_seasons.csv')
events_new_coor = events.copy()[~(events['coordinate_x'].isnull()) & (~events['coordinate_y'].isnull())]

### Adjusting the negative values of coordinate x depending on the team's side

def adjust_negative_value_x(row):
    if row['team_side'] != 'left' and row['team_side'] != 'right':
        row['coordinate_x'] = abs(row['coordinate_x'])
    elif row['team_side'] == 'right' and row['coordinate_x']<0:
        row['coordinate_x'] = abs(row['coordinate_x'])
    return row

events_new_coor = events_new_coor.apply(lambda row: adjust_negative_value_x(row), axis=1)

### Generating lists of game_pk ID for season 2018
### Liste 2018
liste_2018 = []
for string in events_new_coor["datetime"]:
    if string[0:4]=="2018":
        liste_2018.append(string)

### Generate p1 et p2(coordonnées de l'origine du Shot) for 2018 and declare q1 et q2 (goal)
p1_2018 = events_new_coor["coordinate_x"][events_new_coor["datetime"][events_new_coor["datetime"].isin(liste_2018)].index]
p2_2018 = events_new_coor["coordinate_y"][events_new_coor["datetime"][events_new_coor["datetime"].isin(liste_2018)].index]

q1 = 89 # valeur absolue
q2 = 0  # valeur absolue

### calculate Shot_distance

Shot_distance = np.sqrt((abs(q1) - abs(p1_2018))** 2 + (abs(q2) - abs(p2_2018))** 2)

### DataFrame events_new_coor sliced for 2018 season
events_new_coor['game_pk'] = events_new_coor['game_pk'].astype(str)
events_new_coor_2018= events_new_coor[events_new_coor['game_pk'].str.startswith('2018')]

### Nouveau Dataframe pour Saison 2018 : concatenation de  Shot_distances_2018 et du dataset events_new_coor_2018:
Shot_distance_2018 = pd.Series(Shot_distance, index= events_new_coor_2018.index)    ## série correspondant à la distance des tirs effectués des index de la saison 2018
events_new_coor_2018 = pd.concat([events_new_coor_2018, Shot_distance_2018.rename("Shot_distance")], axis=1)  # concaténation de la serie Shot_distance 2018 au dataframe events_new_coor 2018

### define the bins : 20 bins from  minimum value to maximum value
bins = np.linspace(min(distance_tir_effectué), max(distance_tir_effectué), 20)

# add labels if desired
labels = ["1 to 6.1", "6.1 to 11.2", "11.2 to 16.3", "16.3 to 21.4", "21.4 to 26.5", "26.5 to 31.6", "31.6 to 36.7", "36.7 to 41.8", "41.8 to 46.9", "46.9 to 52.0", "52.0 to 57.1", "57.1 to 62.2", "62.2 to 67.4", "67.4 to 72.5", "72.5 to 77.6", "77.6 to 82.7", "82.7 to 87.8", "87.8 to 92.9", "92.9 to 97.98"]

### add the bins to the dataframe :
events_new_coor["bin_Shot_distance"] = pd.cut(events_new_coor_2018["Shot_distance"], bins=bins, labels=labels, include_lowest=True)

### loop qui itere sur chaque bin et filtre le dataframe pour conserver seulement les valeurs qui dans le bin
for interval in bins
    numIterations = 0
    for bin_Shot_distance in events_new_coor_2018["bin_Shot_distance"]
        if interval == bin_Shot_distance
        if events_new_coor_2018['event_type']=="SHOT"
            numIterations += 1

        if valeurs = events_new_coor["bin_Shot_distance"]
        listePourcentage.append(events_new_coor_2018_distance[event_type])

### Pourcentage sur cette iteration et stockage dans une liste

