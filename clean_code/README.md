# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Within this repository, one can perform data loading, EDA, feature engineering, and
model training for predicting customer churn in one command.

## Installation
Please install the dependencies by running the following command:
```pip install -r requirements_py3.8.txt```

## Execution
Please make sure that your working directory is at the project's root. Then run the
following command to execute the functionality offered in the project:
```python clean_code/churn_library.py```

## Tests
The project contains tests for churn_library.py. These can be found in
test_churn_library.py.

Please execute and log the tests with the following command:
```pytest clean_code/test_churn_library.py```

## Files and data description

```
.
├── __init__.py
└── clean_code --> clean code project
    ├── README.md --> README including installation and execution
    ├── churn_library.py --> main file to predict customer churn
    ├── constants.py --> holds constants used in main and test files
    ├── data --> holds data
    │   └── bank_data.csv --> data set used for model training
    ├── images --> eda and results images folder
    │   ├── eda
    │   └── results
    ├── logger.py --> includes logger functionality
    ├── logs --> folder that contains log files
    │   └── churn_library.log
    ├── models --> folder that contains models
    │   ├── logistic_model.pkl
    │   └── rfc_model.pkl
    ├── notebooks --> folder that contains notebooks
    │   ├── Guide.ipynb
    │   └── churn_notebook.ipynb
    ├── pyproject.toml --> toml file that includes ruff and pytest settings
    ├── requirements_py3.8.txt --> installation requirements
    ├── requirements_py3.8_dev.txt --> installation requirements for developers
    └── test_churn_library.py --> unit tests
```
