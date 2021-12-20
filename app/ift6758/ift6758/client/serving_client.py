import json
import requests
import pandas as pd
import logging

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 2000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance_net", "angle_net"]
        self.features = features

        self.current_model = "regression-distance-net-angle-net"

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        X = self.__get_features(X)
        model_info = {'event': X.to_json()}
        r = requests.post(self.base_url+"/predict", json=model_info)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.Series(r.json()['data'])

    def logs(self) -> dict:
        """Get server logs"""
        r = requests.get(self.base_url+"/logs")
        if r.status_code != 200:
            return dict()

        return r.json()

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
        model_info = {'workspace': workspace, 'model': model, 'version': version}
        r = requests.post(self.base_url+"/download_registry_model", json=model_info)
        if r.status_code == 200:
            self.current_model = model

    def __get_features(self, X):
        if self.current_model == "regression-distance-net":
            self.features = ["distance_net"]
        elif self.current_model == "regression-angle-net":
            self.features = ["angle_net"]
        elif self.current_model == "regression-distance-net-angle-net":
            self.features = ["distance_net", "angle_net"]
        elif self.current_model == "xgb-tuning2":
            scale_features = ['coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'time_since_pp_started',
                              'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                              'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                              'shot_last_event_delta', 'shot_last_event_distance', 'Change_in_shot_angle', 'Speed']
            scaler = StandardScaler()
            X[scale_features] = scaler.fit_transform(X[scale_features])
            X = X.drop(columns='is_goal')
            self.features = X.columns.tolist()
        elif self.current_model == "xgb-lasso":
            scale_features = ['coordinate_x', 'coordinate_y', 'distance_net', 'angle_net', 'time_since_pp_started',
                              'previous_event_time_seconds', 'current_time_seconds', 'current_friendly_on_ice',
                              'current_opposite_on_ice', 'previous_event_x_coord', 'previous_event_y_coord',
                              'shot_last_event_delta', 'shot_last_event_distance', 'Change_in_shot_angle', 'Speed']
            scaler = StandardScaler()
            X[scale_features] = scaler.fit_transform(X[scale_features])
            self.features = ['coordinate_x', 'distance_net', 'previous_event_time_seconds',
                'current_opposite_on_ice', 'shot_last_event_delta', 'Rebound',
                'Speed', ('Backhand',), ('Deflected',), ('Slap Shot',),
                ('Snap Shot',), ('Tip-In',), ('Wrap-around',), (1,),
                (3,), (4,), ('left',), ('FACEOFF',), ('GIVEAWAY',),
                ('HIT',), ('TAKEAWAY',)
            ]
        return X[self.features]
