from client import ServingClient
from game_client import GameClient
import pandas as pd


client = ServingClient()
game_client = GameClient()

events = game_client.ping_game(game_id=2021020329)

client.download_registry_model(workspace="jaihon", model="xgb-lasso", version="1.0.0")
print(client.predict(events))

client.download_registry_model(workspace="jaihon", model="xgb-tuning2", version="1.0.0")
print(client.predict(events))

client.download_registry_model(workspace="jaihon", model="regression-distance-net", version="1.0.0")
print(client.predict(events))

client.download_registry_model(workspace="jaihon", model="regression-distance-net-angle-net", version="1.0.0")
print(client.predict(events))

client.download_registry_model(workspace="jaihon", model="regression-angle-net", version="1.0.0")
print(client.predict(events))
