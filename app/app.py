"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from typing import List
from flask import Flask, Response, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from custom_exceptions import EmptyLogs



LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)

# class EmptyLogs(Exception):
#     status_code = 400

#     def __init__(self, message, status_code=None, payload=None):
#         Exception.__init__(self)
#         self.message = message
#         if status_code is not None:
#             self.status_code = status_code
#         self.payload = payload

#     def log_error(self):
#         logging.error(f"Empty Log Error: {self.to_dict()}")

#     def to_dict(self):
#         rv = dict(self.payload or ())
#         rv['message'] = self.message
#         rv['status_code'] = self.status_code
#         return rv


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    pass


@app.errorhandler(EmptyLogs)
def handle_foo_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    error.log_error()

    return response

@app.route("/logs", methods=["GET"])
def logs():
    """
        Reads data from the log file and returns them as the response.
    """
    data: List = []
    try:
        with open('flask.log') as f:
            # Read the log file
            lines = f.readlines()

            # Build the response data
            for line in lines:
                r = line.split('\t\t')
                data.append(r[0])

            app.logger.info('Log file successfully read.')
    except Exception as e:
        app.logger.error('Failed to read Log file: '+ str(e))

    # # Verify that the logs is not empty
    # if not data:
    #     raise EmptyLogs('Logs are empty.', status_code=404)

    return jsonify(data), 200 # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.
    # eg: app.logger.info(<LOG STRING>)

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
