
import pandas as pd
import numpy as np
import timeit
import os
import json 

import pickle
from pathlib import Path


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = Path().cwd() / config['output_folder_path'] / "finaldata.csv"
test_data_path = Path().cwd() / config['test_data_path'] / "testdata.csv"
prod_deployment_path = Path().cwd() / config['prod_deployment_path']


##################Function to get model predictions
def model_predictions():
    with open(prod_deployment_path / "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(test_data_path)

    y_test = df["exited"]
    X_test = df.drop(["corporation", "exited"], axis=1)

    preds = model.predict(X_test)

    return y_test, preds


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)

    numeric_cols = [i for i in df if df[i].dtypes == int]

    output_ls = []

    for i in numeric_cols:
        output_ls.append(
            [
                i,
                round(df[i].mean(), 3),
                round(df[i].median(), 3),
                round(df[i].std(), 3),
            ]
        )

    return output_ls


def check_missing_data():
    df = pd.read_csv(dataset_csv_path)
    missing_data = df.isna().mean() * 100

    missing_data_list = []
    for col, pct_missing in zip(missing_data.index, missing_data.tolist()):
        missing_data_list.append([col, pct_missing])

    return missing_data_list


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    ingestions_time = timeit.timeit(
        f'{os.system("python ingestion.py")}'
    )

    training_time = timeit.timeit(
        f'{os.system("python training.py")}'
    )

    return [ingestions_time, training_time]


##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    output_ls = []

    with open("requirements.txt", "r") as requirements_file:
        for line in requirements_file:
            line_parts = line.strip().split("==")
            package_name = line_parts[0]
            current_version = line_parts[1]
            #or: current_version = os.popen(f"pip show {package_name} | grep Version").read().strip().split(": ")[1]
            newest_version = (
                os.popen(f"pip index versions {package_name} | grep LATEST:")
                .read()
                .strip()
                .split(":")[1]
                .replace(" ", "")
            )

            output_ls.append(
                [
                    package_name,
                    current_version,
                    newest_version,
                ]
            )

    return output_ls


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()
