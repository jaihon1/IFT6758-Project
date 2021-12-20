"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import logging
from typing import List
from flask import Flask, jsonify, request, abort
import pandas as pd
import joblib


from comet_ml import API


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


# Move this to env variables!!
COMET_API_KEY = os.environ.get("COMET_API_KEY", "ZqM4liL9boT3pGhQWAP5Bj1xD")
COMET_DEFAUTL_MODEL_WORKSPACE = os.environ.get("COMET_DEFAUTL_MODEL_WORKSPACE", 'jaihon')
COMET_DEFAULT_MODEL_NAME = os.environ.get("COMET_DEFAULT_MODEL_NAME", 'regression-distance-net-angle-net')
COMET_DEFAULT_MODEL_VERSION = os.environ.get("COMET_DEFAULT_MODEL_VERSION", '1.0.0')

# Set current model to default
CURRENT_MODEL_WORKSPACE = COMET_DEFAUTL_MODEL_WORKSPACE
CURRENT_MODEL_NAME = COMET_DEFAULT_MODEL_NAME
CURRENT_MODEL_VERSION = COMET_DEFAULT_MODEL_VERSION


app = Flask(__name__)
api = API(api_key=COMET_API_KEY)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # Setup logging
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
    )

    # Download a Registry Model as a default model
    try:
        api.download_registry_model(CURRENT_MODEL_WORKSPACE, CURRENT_MODEL_NAME, CURRENT_MODEL_VERSION, output_path="./models", expand=True)

        app.logger.info(f"Downloaded default model from Registry: {CURRENT_MODEL_WORKSPACE}/{CURRENT_MODEL_NAME}/{CURRENT_MODEL_VERSION}")

    except Exception as e:
        app.logger.error(f"Failed to download default model: {e}")


@app.route("/logs", methods=["GET"])
def logs():
    """
    Reads data from the log file and returns them as the response.
    """
    data: List = []
    try:
        with open(LOG_FILE) as f:
            # Read the log file
            lines = f.readlines()

            # Build the response data
            for line in lines:
                r = line.split('\t\t')
                data.append(r[0])

            app.logger.info('Log file successfully read')

    except Exception as e:
        # Log the error
        app.logger.error('Failed to read Log file: '+ str(e))

        # # Return the error
        # return abort(404, description="Failed to read Log file")

    # Build the response
    response = {
        "data": data,
        "success": True
    }
    return jsonify(response), 200


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
        }

    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Get the workspace, model and version from the request
    workspace = json['workspace']
    model_name = json['model']
    model_version = json['version']

    # Set global variables
    global CURRENT_MODEL_WORKSPACE
    global CURRENT_MODEL_NAME
    global CURRENT_MODEL_VERSION

    if workspace == COMET_DEFAUTL_MODEL_WORKSPACE and model_name == COMET_DEFAULT_MODEL_NAME and model_version == COMET_DEFAULT_MODEL_VERSION:
        # Use default model
        # Set current model to default
        CURRENT_MODEL_WORKSPACE = COMET_DEFAUTL_MODEL_WORKSPACE
        CURRENT_MODEL_NAME = COMET_DEFAULT_MODEL_NAME
        CURRENT_MODEL_VERSION = COMET_DEFAULT_MODEL_VERSION

        app.logger.info(f'No change required for model. Using default model from Registry: {CURRENT_MODEL_WORKSPACE}/{CURRENT_MODEL_NAME}/{CURRENT_MODEL_VERSION}')

    else:
        try:
            # Download the requested model
            api.download_registry_model(workspace, model_name, model_version, output_path="./models", expand=True)

            # Set current model to requested model
            CURRENT_MODEL_WORKSPACE = workspace
            CURRENT_MODEL_NAME = model_name
            CURRENT_MODEL_VERSION = model_version

            # Log the changes
            app.logger.info(f"Changed model. Downloaded model from Registry: {CURRENT_MODEL_WORKSPACE}/{CURRENT_MODEL_NAME}/{CURRENT_MODEL_VERSION}")

        except Exception as e:
            app.logger.error(f"Failed to download requested model from Registry: {e}")

            # Return the error
            return abort(404, description="Failed to download model from Registry. Keeping currently loaded model.")

    # Build the response
    response = {
        'data': None,
        'success': True
    }
    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    {
        event: (required),
    }

    Returns predictions
    """

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    event = json['event']

    # Load the model
    model = joblib.load('models/'+CURRENT_MODEL_NAME.replace('-', '_')+'.joblib')

    # Predict
    try:
        predictions = model.predict_proba(pd.read_json(event))[:, 1]

    except Exception as e:
        app.logger.error(f"Failed to predict with current model {CURRENT_MODEL_WORKSPACE}/{CURRENT_MODEL_NAME}/{CURRENT_MODEL_VERSION}: {e}")

        # Return the error
        return abort(404, description="Failed to predict :(")

    # Build the response
    response = {
        "data": list(predictions.astype(str)),
        "success": True
    }

    app.logger.info(response)
    return jsonify(response)
