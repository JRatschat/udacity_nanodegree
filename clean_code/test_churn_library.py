"""
File that tests functions used in main file (churn_library.py).

Author: Jonathan Ratschat
Date: 21.03.2022
"""
import pandas as pd
import pytest

from churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
)
from constants import (
    data_path,
    eda_imgs_ls,
    image_path,
    model_path,
    models_ls,
    train_imgs_ls,
)
from logger import load_logger

logging = load_logger("tests.log")


def test_import():
    """Test data import."""
    try:
        df = import_data(data_path / "bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.exception("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.exception(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


@pytest.fixture()
def load_data():
    return import_data(data_path / "bank_data.csv")


def test_eda(load_data):
    """Test perform eda function."""
    perform_eda(load_data)
    for eda_img in eda_imgs_ls:
        try:
            assert (image_path / "eda" / eda_img).is_file()
            logging.info(f"Testing perform_eda for {eda_img}: SUCCESS")
        except AssertionError as err:
            logging.exception(f"Testing perform_eda: {eda_img} is missing")
            raise err


def test_encoder_helper():
    """Test encoder helper."""
    try:
        test_df = pd.DataFrame(
            {
                "Churn": [1, 1, 0, 1, 0, 1],
                "Gender": ["M", "M", "F", "M", "F", "B"],
                "Card": ["Credit", "Credit", "Credit", "Debit", "Credit", "Debit"],
            }
        )
        category_lst = ["Gender", "Card"]
        test_df = encoder_helper(test_df, category_lst)
        assert all(test_df["Gender_Churn"] == pd.Series([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
        assert all(test_df["Card_Churn"] == pd.Series([0.5, 0.5, 0.5, 1.0, 0.5, 1.0]))
        logging.info(
            "Testing encoder_helper: SUCCESS. The churn outputs are calculated as"
            " expected"
        )
    except AssertionError as err:
        logging.exception(
            "Testing encoder_helper: The result is different from the expected results"
        )
        raise err
    try:
        assert test_df.shape[0] == 6
        assert test_df.shape[1] == 5
        logging.info(
            "Testing encoder_helper: SUCCESS. The test data frame has the right amount"
            " of columns and rows"
        )
    except AssertionError as err:
        logging.exception(
            "Testing encoder_helper: The file doesn't appear to have rows and columns"
        )
        raise err


def test_perform_feature_engineering(load_data):
    """Test perform_feature_engineering."""
    X_train, X_test, y_train, y_test = perform_feature_engineering(load_data)
    try:
        assert len(X_train) == len(y_train)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS. X and y train sets have the"
            " same lenght"
        )
    except AssertionError as err:
        logging.exception(
            "Testing perform_feature_engineering: X and y train sets do not have the"
            " same lenght"
        )
        raise err
    try:
        assert len(X_test) == len(y_test)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS. X and y test sets have the"
            " same lenght"
        )
    except AssertionError as err:
        logging.exception(
            "Testing perform_feature_engineering: X and y test sets do not have the"
            " same lenght"
        )
        raise err


def test_train_models(load_data):
    """Test train_models."""
    X_train, X_test, y_train, y_test = perform_feature_engineering(load_data)
    train_models(X_train, X_test, y_train, y_test)
    for train_img in train_imgs_ls:
        try:
            assert (image_path / "results" / train_img).is_file()
            logging.info(f"Testing train_models for {train_img}: SUCCESS")
        except AssertionError as err:
            logging.exception(f"Testing train_models: {train_img} is missing")
            raise err
    for model in models_ls:
        try:
            assert (model_path / model).is_file()
            logging.info(f"Testing train_models for {model}: SUCCESS")
        except AssertionError as err:
            logging.exception(f"Testing train_models: {model} is missing")
            raise err
