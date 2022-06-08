#!/usr/bin/env python3
# coding: utf-8

import os
import json
from regex import F
from scipy import rand
import tensorflow as tf
import numpy as np
import pandas as pd
import flask
from flask_cors import CORS


# custom module imports
# import predict
import neural_net as nn
import read_h5 as read
import preprocessing as pp
import features

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)
model = None


def load_model():
    global lookupDF
    global song_file_map
    global column_maps
    global max_list
    global model
    global scaler
    global graph
    global probDF

    # Load model
    graph = tf.compat.v1.get_default_graph()

    model = nn.load_model('./model/working/std')

    # # Load preprocessing dependencies
    # with open('./data/song-file-map.json', 'r') as f:
    #     song_file_map = json.load(f)
    # with open('./model/working/preprocessing/maps.json', 'r') as f:
    #     column_maps = json.load(f)
    # with open('./model/working/preprocessing/max_list.json', 'r') as f:
    #     max_list = json.load(f)

    # scaler = joblib.load('./model/working/preprocessing/robust.scaler')

    # # Load song ID lookup for frontend
    # lookupDF = pd.read_hdf('./frontend/data/lookup.h5', 'df')

    # # Model predictions for comparison
    # probDF = pd.read_pickle('./data/model_prob.pkl')


def process_metadata_list(col):
    x_map = column_maps[col.name]
    max_len = max_list[col.name]
    col = col.apply(lambda x: pp.lookup_discrete_id(x, x_map))
    col = col.apply(lambda x: np.pad(x, (0, max_len - x.shape[0]), 'constant'))
    xx = np.stack(col.values)
    return xx


def preprocess_predictions(df):
    print('Vectorizing dataframe...')
    for col in df:
        if df[col].dtype == 'O':
            if type(df[col].iloc[0]) is str:
                xx = pp.lookup_discrete_id(df[col], column_maps[col])
                xx = xx.reshape(-1, 1)
            elif col.split('_')[0] == 'metadata':
                xx = process_metadata_list(df[col])
            else:
                xx = pp.process_audio(df[col])

        else:
            xx = df[col].values[..., None]

        # Normalize each column
        xx = xx / (np.linalg.norm(xx) + 0.00000000000001)

        try:
            output = np.hstack((output, xx))
        except NameError:
            output = xx

    return output


def get_song_features(file_data):
    result = features.get_features(file_data)

    return pd.DataFrame.from_records([result])


def _build_cors_preflight_response():
    response = flask.make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route("/analyze", methods=["POST"])
def analyze():
    if flask.request.method == "OPTIONS":  # CORS preflight
        return _build_cors_preflight_response()

    data = {"success": False, 'verdict': None, 'probability': None}

    # POST requests
    if flask.request.method == "POST":
        file_data = flask.request.args.get('audio_file')
        f = features.get_features(file_data)
        X = pp.preprocess_features(f)

        X = np.loadtxt('src/aaa.csv', delimiter=',')
        X = np.random.rand(1, 564)

        prob = model.predict(X)

        verdict = bool(prob.argmax())
        print(prob.shape)
        print(verdict, prob[0][1])
        data['probability'] = str(prob[0][1])
        data['verdict'] = verdict

    # indicate that the request was a success
    data["success"] = True

    response = flask.jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")

    return response


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(" * Starting Flask server and loading Keras model...")
    print(" * Please wait until server has fully started")

    # Load model and dependencies
    load_model()

    print(' * Server is active')
    # Run app
    app.run(host='0.0.0.0', port=5001)
