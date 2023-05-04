import json
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import f1_score

import training
import scoring
import deployment
import diagnostics
import reporting


with open('config.json','r') as f:
    config = json.load(f)

input_path = Path().cwd() / config['input_folder_path']
output_path = Path().cwd() / config['output_folder_path']
prod_deployment_path = Path().cwd() / config['prod_deployment_path']
model_path = Path().cwd() / config['output_model_path']

def fullprocess():
##################Check and read new data
#first, read ingestedfiles.txt
    with open(output_path / "ingestedfiles.txt") as f:
        file = f.read()
        file_list = file.split("\n")

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    files = list(input_path.glob("*.csv"))
    missing_files = [i.name for i in files if i.name not in file_list]

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
    if missing_files:
        return None

    os.system("python ingestion.py")

    with open(prod_deployment_path / "latestscore.txt") as f:
        f1_latest = float(f.read())

    os.system("python scoring.py")

    with open(model_path / "latestscore.txt") as f:
        f1_new = float(f.read())

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
    if f1_new <= f1_latest:
        return None

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    os.system("python deployment.py")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
    os.system("python diagnostics.py")
    os.system("python reporting.py")

if __name__ == "__main__":
    fullprocess()
