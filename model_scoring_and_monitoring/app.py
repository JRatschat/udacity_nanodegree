from flask import Flask, session, jsonify, request

from diagnostics import (
    model_predictions,
    dataframe_summary,
    check_missing_data,
    execution_time,
    outdated_packages_list,
)
from scoring import score_model
import json
import os

from pathlib import Path


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = Path().cwd() / config['output_folder_path'] / "finaldata.csv"

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def prediction():
    #call the prediction function you created in Step 3
    _, preds = model_predictions()
    return jsonify(preds.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    return str(score_model())

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    #check means, medians, and modes for each column
    return jsonify(dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    return jsonify(check_missing_data(), execution_time(), outdated_packages_list())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
