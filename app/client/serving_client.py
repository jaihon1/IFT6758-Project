import json
import requests
import pandas as pd
import logging


### logger object
#logging.basicConfig(filename = "E", level = logging.DEBUG)
logger = logging.getLogger(__name__)

###  Test the logger
logger.info("Our first mesage")
print(logger.level)

data = {"calories": [420, 380, 390],"duration": [50, 40, 45]}
X = pd.DataFrame(data)
###
#def get_input_features_df():
#    return

class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        #X = get_input_features_df()
        #print(X)
        r = (f"{self.base_url}/predict", json.loads(X.to_json()))
        #print(r.json())
        #raise NotImplementedError("TODO: implement this function")

    def logs(self) -> dict:
        """Get server logs"""
        r = (f"{self.base_url}/logs", json.loads(X.to_json()))
        #print(r.json())
        #raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        r = (f"{self.base_url}/download_registry_model", json.loads(X.to_json()))
        #print(r.json())
        #raise NotImplementedError("TODO: implement this function")


Objet = ServingClient()
Objet.predict(X)
Objet.logs()
Objet.download_registry_model("jaihon", "xgb-all-features", 1)