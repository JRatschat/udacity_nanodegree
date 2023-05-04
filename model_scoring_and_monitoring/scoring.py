from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from pathlib import Path


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = Path().cwd() / config['output_folder_path'] / "finaldata.csv"
test_data_path = Path().cwd() / config['test_data_path'] / "testdata.csv"
model_path = Path().cwd() / config['output_model_path']



#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(model_path / "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(test_data_path)

    y_test = df["exited"]
    X_test = df.drop(["corporation", "exited"], axis=1)

    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds)

    with open(model_path / "latestscore.txt", "w") as f:
        print(f1, file=f)

    return f1


if __name__ == "__main__":
    score_model()
