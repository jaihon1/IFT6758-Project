import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("C:/Users/samib/PycharmProjects/pythonGenerateCSV/games_data/games_data_all_seasons.csv")


Series_of_Goals = df["event_type"][df["event_type"].isin(['GOAL'])]
Number_of_Goals = Series_of_Goals.count()
#print(Number_of_Goals)

liste_2016 = []
for integer in df["game_pk"]:
    if str(integer)[0:4]=="2016":
        #print(integer)
        liste_2016.append(integer)
Number_of_shots_per_Shot_types = df["shot_type"][df["game_pk"][df["game_pk"].isin(liste_2016)].index][df["event_type"].isin(['SHOT'])].value_counts()
Number_of_shots_per_Shot_types
Number_of_Goals_per_Shot_types = df["shot_type"][df["game_pk"][df["game_pk"].isin(liste_2016)].index][df["event_type"].isin(['GOAL'])].value_counts()
Number_of_Goals_per_Shot_types

# Series_of_Shots = df["event_type"][df["event_type"].isin(['SHOT'])]
# Number_of_shots_per_Shot_types = df["shot_type"][0:46765][df["event_type"].isin(['SHOT'])].value_counts()
# Number_of_Goals_per_Shot_types = df["shot_type"][0:46765][df["event_type"].isin(['GOAL'])].value_counts()

#print(Number_of_shots_per_Shot_types)
#print(Number_of_Goals_per_Shot_types)
new_index= ['Wrist Shot', 'Slap Shot', 'Snap Shot', 'Backhand', 'Tip-In', 'Deflected', 'Wrap-around']
Number_of_shots = Number_of_shots_per_Shot_types.reindex(new_index)
Number_of_Goals = Number_of_Goals_per_Shot_types.reindex(new_index)
print(Number_of_shots)
print(Number_of_Goals)

# sns.barplot(Number_of_shots_per_Shot_types.index, Number_of_shots_per_Shot_types.values, alpha=0.5)
# plt.title('Number of shots per number of shot types over all teams in the 2016 season')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Types of Shots', fontsize=12)
#
# plt.show()
#
# sns.barplot(Number_of_Goals_per_Shot_types.index, Number_of_Goals_per_Shot_types.values, alpha=0.5)
# plt.title('Number of Goals per number of shot types over all teams in the 2016 season')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Types of Shots', fontsize=12)
# plt.show()

#### Overlap of the SHOTS and GOALS : BARPLOT
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax2 = ax.twinx()
ax.set_title('Shots and Goals per number of shot types over all teams in the 2016 season')
ax.set_xlabel('Types of Shots', fontsize=12)
sns.barplot(Number_of_shots.index, Number_of_shots_per_Shot_types.values, alpha=0.4)
sns.barplot(Number_of_Goals.index, Number_of_Goals_per_Shot_types.values, alpha=0.4)
ax2.set_ylim(0, 25000)
ax.set_ylim(0, 25000)
ax.set_ylabel('Number of Occurrences')
#ax2.set_ylabel('Number of Occurrences')
ax2.get_yaxis().set_visible(False)
plt.show()

### Most dangerous type of Shots
Dangerous_Shots = Number_of_Goals / Number_of_shots * 100
print(Dangerous_Shots)
sns.barplot(Dangerous_Shots.index, Dangerous_Shots.values, alpha=0.5)
plt.title('Most dangerous type of shots in the 2016')
plt.ylabel('Percentage of Goals on Shots', fontsize=12)
plt.xlabel('Types of Shots', fontsize=12)
plt.show()



#plt.savefig("overlapping_histograms_with_matplotlib_Python.png")

# ax2 = ax.twinx()
# ax.grid()
# ax1.plot(X1,Y)
# ax1.set_ylim([0, 1.0])


# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(X2)









#twin_axes=axes.twinx().twiny()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(time, Swdown, '-', label = 'Swdown')
#ax.plot(time, Rn, '-', label = 'Rn')
#ax2 = ax.twinx()
#ax2.plot(time, temp, '-r', label = 'temp')
#ax.legend(loc=0)
#ax.set_xlabel("Time (h)")
#ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
#ax2.set_ylabel(r"Temperature ($^\circ$C)")
#ax2.set_ylim(0, 35)
#ax.set_ylim(-20,100)
#plt.show()

# plt.figure(figsize=(8,6))
# ax.set_title('Types of Shots', fontsize=12)
# ax.set_xlabel('Types of Shots', fontsize=12)
#
# ax.plot(gdp['date'], gdp[' GDP Per Capita (US $)'], color='green', marker='x')
# ax2.plot(gdp['date'], gdp[' Annual Growth Rate (%)'], color='red', marker='o')
#
# ax.set_ylabel('Number of Occurrences', fontsize=12)
# ax2.set_ylabel('Annual Growth Rate (%)')
# ax.legend(['GDP Per Capita (US $)'])
# ax2.legend(['Annual Growth Rate (%)'], loc='upper center')
# ax.set_xticks(gdp['date'].dt.date)
# ax.set_xticklabels(gdp['date'].dt.year, rotation=90)
# ax.yaxis.grid(color='lightgray', linestyle='dashed')
# plt.tight_layout()
# plt.show()

import pandas as pd


# Plot the total crashes
# f, ax = plt.subplots(figsize=(10, 5))
# plt.xticks(rotation=90, fontsize=10)
#
# plt.bar(height="total", x="abbrev", data=Number_of_shots_per_Shot_types, label="Total", color="lightgray")
# plt.bar(height="total", x="abbrev", data=Number_of_Goals_per_Shot_types, label="Total", color="red")


# sns.despine(left=True, bottom=True)


#shot_type = df['shot_type'][0:46765]
#shot_type = shot_type.to_numpy()
#print(shot_type)
#bins = ["Deflected", "Wrap-around", "Wrist Shot", "Backhand", "Slap Shot", "Tip-In", "Snap Shot"]
#plt.hist(shot_type)

#plt.title("Multiple Histograms with Matplotlib")
#plt.xlim([0,100])
#plt.ylim([0,18000])
#plt.show()

#plt.legend(loc='upper right')
#plt.savefig("overlapping_histograms_with_matplotlib_Python.png")

#plt.figure(figsize=(8,6))
#plt.hist(data["shot_type"], team_id bins=2, alpha=0.5, label="data1")
#plt.hist(data2, bins=2, alpha=0.5, label="data2")

#df.plot(kind='hist')
#df.plot.hist()


