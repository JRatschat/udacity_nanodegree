import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from pathlib import Path

from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = Path().cwd() / config['output_folder_path'] / "finaldata.csv"
model_path = Path().cwd() / config['output_model_path']

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    y_test, preds = model_predictions()

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(model_path / "confusionmatrix.png")


if __name__ == '__main__':
    score_model()
