#%%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
data_path = os.path.join(os.environ.get('PATH_DATA'), "games_data_all_seasons.csv")
data = pd.read_csv(data_path)

data['game_pk'] = data['game_pk'].apply(lambda i : str(i))
data = data[~data['game_pk'].str.startswith('2019')]

#%%
# plot histogram of goal (empty net and non-empty net) according to distance from net
data_goals = data.loc[data['is_goal'] == 1]

hist_distance_goals = sns.histplot(data=data_goals, x='distance_net', hue='empty_net', bins=15, multiple='layer', element='step',
                                   hue_order=[1, 0], palette=['C1', 'C0'])

plt.xlabel('Distance from net (feet)')
plt.ylabel('Number of goals')
plt.legend(labels=['non-empty net', 'empty net'])
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# plot histogram of goal or no goal according to distance from net
sns.set_theme()
hist_distance = sns.histplot(data=data, x='distance_net', hue='is_goal', bins=15, multiple='layer', element='step',
                             hue_order=[1, 0], palette=['C1', 'C0'])

plt.xlabel('Distance from net (feet)')
plt.ylabel('Number of shots (goal or no goal)')
plt.legend(labels=['no goal', 'goal'])
plt.tight_layout()
plt.show()

#%%
# plot histogram of goal or no goal according to angle from net
sns.set_theme()
hist_angle = sns.histplot(data=data, x='angle_net', hue='is_goal', bins=15, multiple='layer', element='step',
                          hue_order=[1, 0], palette=['C1', 'C0'])

plt.xlabel('Angle from net (degrees)')
plt.ylabel('Number of shots (goal or no goal)')
plt.legend(labels=['no goal', 'goal'])
plt.tight_layout()
plt.show()

#%%
# joint plot of distance vs angle
joint_distance_angle = sns.jointplot(data=data, x='distance_net', y='angle_net')
plt.tight_layout()
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
plt.xlabel('angle from net (degrees)')
plt.ylabel('goal rate')
plt.xticks(fontsize=20, rotation=70)
plt.tight_layout()
plt.show()


