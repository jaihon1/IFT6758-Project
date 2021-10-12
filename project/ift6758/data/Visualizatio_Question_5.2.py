import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Users/samib/PycharmProjects/pythonGenerateCSV/games_data/games_data_all_seasons.csv")
### saison 2018
p1_2018 = df[coordinate_x][df[datetime][133905:219844]]
p2_2018 = df[coordinate_y][df[datetime][133905:219844]]

### saison 2019
p1_2019 = df[coordinate_x][df[datetime][219844:295715]]
p2_2019 = df[coordinate_y][df[datetime][219844:295715]]

### saison 2020
p1_2020 = df[coordinate_x][df[datetime][295715:353449]]
p2_2020 = df[coordinate_y][df[datetime][295715:353449]]

q1 = -89.0 or +89.0
q2 = 0


distance = math.sqrt((q1 - p1) ** 2 + (q2 - p2) ** 2)


chance_de_buts = Number_of_Goals / Number_of_shots * 100