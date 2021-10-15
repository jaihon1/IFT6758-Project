import pandas as pd
import os as os
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
file_path = os.path.join(os.path.dirname(__file__), 'games_data_all_seasons.csv')

df = pd.read_csv(file_path, index_col='game_pk')
df = df[df.columns[1:]]
print(df.head(10))