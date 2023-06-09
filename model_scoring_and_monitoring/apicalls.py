import requests
import json
import pandas as pd

from pathlib import Path


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)
model_path = Path().cwd() / config['output_model_path']
test_data_path = (Path().cwd() / config['test_data_path'] / "testdata.csv").as_posix()

#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"

#Call each API endpoint and store the responses
response1 = requests.post(URL  + "prediction", json={"path": test_data_path})
response2 = requests.get(URL + "scoring")
response3 = requests.get(URL + "summarystats")
response4 = requests.get(URL + "diagnostics")

#combine all API responses
#write the responses to your workspace
with open(model_path / "apireturns2.txt", "w") as f:
    print(response1.json(), file=f)
    print(response2.json(), file=f)
    print(response3.json(), file=f)
    print(response4.json(), file=f)
