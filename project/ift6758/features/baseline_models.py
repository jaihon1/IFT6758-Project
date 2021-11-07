import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('project/ift6758/data/games_data/games_data_all_seasons.csv')

data = data[~(data['distance_net'].isna())]

data['game_pk'] = data['game_pk'].apply(lambda i: str(i))

train, test = data[~data['game_pk'].str.startswith('2019')], data[data['game_pk'].str.startswith('2019')]
#%%
train, valid = train_test_split(train, test_size=0.2)

#%%
clf = LogisticRegression()
clf.fit(train['distance_net'].array.reshape(-1, 1), train['is_goal'])

#%%
# Evaluate default logistic regression
pred = clf.predict(valid['distance_net'].array.reshape(-1, 1))
accuracy_default = np.where(pred == valid['is_goal'], 1, 0).sum()/len(pred)
confusion_mat = confusion_matrix(valid['is_goal'], pred)

class1 = valid['is_goal'].sum()/len(valid)
class0 = (len(valid) - class1)/len(valid)
