import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from pathlib import Path


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = Path().cwd() / config['input_folder_path']
output_folder_path = Path().cwd() / config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    files = list(input_folder_path.glob("*.csv"))
    df_list = []
    for file_path in files:
        df_temp = pd.read_csv(file_path)
        df_list.append(df_temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df.drop_duplicates()

    df.to_csv(output_folder_path / "finaldata.csv", index=False)

    with open(output_folder_path / "ingestedfiles.txt", "w") as f:
        for file_path in files:
            print(file_path.name, file=f)


if __name__ == "__main__":
    merge_multiple_dataframe()
