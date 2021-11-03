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

# %%
# goal rate vs distance
bins = np.linspace(data['distance_net'].min(), data['distance_net'].max(), 15)
label = ['{min:.1f}-{max:.1f}'.format(min=bins[i], max=bins[i + 1]) for i in range(len(bins) - 1)]

data['bin_distance'] = pd.cut(data['distance_net'], bins, labels=label)
goal_rate_by_distance = data.groupby(by=['bin_distance']).apply(lambda g: (g['is_goal'] == 1).sum() / (len(g)))

fig = plt.figure(figsize=(12, 10))
sns.set(font_scale=2)

sns.barplot(data=data, x=label, y=goal_rate_by_distance)
plt.xlabel('distance from net (feet)')
plt.ylabel('goal rate')
plt.xticks(fontsize=20, rotation=70)
plt.tight_layout()
plt.show()

# %%
# goal rate vs angle
bins = np.linspace(data['angle_net'].min(), data['angle_net'].max(), 15)
label = ['{min:.1f}-{max:.1f}'.format(min=bins[i], max=bins[i + 1]) for i in range(len(bins) - 1)]

data['bin_angle'] = pd.cut(data['angle_net'], bins, labels=label)
goal_rate_by_distance = data.groupby(by=['bin_angle']).apply(lambda g: (g['is_goal'] == 1).sum() / (len(g)))

fig = plt.figure(figsize=(12, 10))
sns.set(font_scale=2)

sns.barplot(data=data, x=label, y=goal_rate_by_distance)
plt.xlabel('angle from net (feet)')
plt.ylabel('goal rate')
plt.xticks(fontsize=20, rotation=70)
plt.tight_layout()
plt.show()
