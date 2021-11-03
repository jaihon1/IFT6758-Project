import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%

data = pd.read_csv('./project/ift6758/data/games_data/games_data_all_seasons.csv')

#%%
# plot histogram of goal or no goal according to distance from net
hist_distance = sns.histplot(data=data, x='distance_net', hue='is_goal', bins=15, multiple='layer', element='step',
                          hue_order=[1, 0], palette=['C1', 'C0'])

plt.xlabel('Distance from net (feet)')
plt.ylabel('Number of shots (goal or no goal)')
plt.legend(labels=['goal', 'no goal'])
plt.show()

#%%
# plot histogram of goal or no goal according to angle from net
hist_angle = sns.histplot(data=data, x='angle_net', hue='is_goal', bins=15, multiple='layer', element='step',
                             hue_order=[1, 0], palette=['C1', 'C0'])

plt.xlabel('Angle from net (degrees)')
plt.ylabel('Number of shots (goal or no goal)')
plt.legend(labels=['goal', 'no goal'])
plt.show()

#%%
# joint plot of distance vs angle
joint_distance_angle = sns.jointplot(data=data, x='distance_net', y='angle_net')
plt.show()