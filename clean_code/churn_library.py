"""
Main file to load data, perform eda and feature engineering, and train models.

Author: Jonathan Ratschat
Date: 21.03.2022
"""
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import (
    categorical_cols_lst,
    data_path,
    image_path,
    keep_cols_lst,
    model_path,
    param_grid,
)
from logger import load_logger

logging = load_logger("churn_library.log")


def import_data(pth: Path) -> pd.DataFrame:
    """Returns dataframe for the csv found at pth and create churn column.

    Args:
        pth: a path to the csv.

    Returns:
        df with included churn column.
    """
    logging.info("Starting to load data set")
    df = pd.read_csv(pth)

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    logging.info("Data set successfully loaded")

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Perform eda on df and save figures to images folder.

    Args:
        df: pandas dataframe

    Returns:
        None
    """
    logging.info("Starting to perform EDA")

    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig(image_path / "eda/churn_histogram.png")

    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig(image_path / "eda/customer_age_histogram.png")

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(image_path / "eda/marital_status_value_counts_norm_bar.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(image_path / "eda/total_trans_density.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(image_path / "eda/correlation_heatmap.png")

    logging.info("EDA successfully conducted")


def encoder_helper(df: pd.DataFrame, category_lst: list) -> pd.DataFrame:
    """Helper function to turn each categorical column into a new column.

    Each categorical column is transformed in proportion with churn.

    Args:
            df: Data frame that contains categorical columns.
            category_lst: List of columns that contain categorical features.

    Returns:
            df: Transformed data frame.
    """
    logging.info(
        f"""
        Starting to encode categorical variables for the following columns:
        {category_lst}
        """
    )
    for category in category_lst:
        col_lst = []
        col_groups = df.groupby(category).mean()["Churn"]
        for val in df[category]:
            col_lst.append(col_groups.loc[val])
        df[f"{category}_Churn"] = col_lst

    logging.info("Encoding successfully conducted")

    return df


def perform_feature_engineering(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Args:
        df: pandas dataframe

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    logging.info("Starting to perform feature engineering")
    X = pd.DataFrame()
    y = df["Churn"]

    df = encoder_helper(df, categorical_cols_lst)

    X[keep_cols_lst] = df[keep_cols_lst]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    logging.info("Successfully performed feature engineering")

    return X_train, X_test, y_train, y_test


def classification_report_image(
    model_name: str,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train_preds_lr: np.ndarray,
    y_train_preds_rf: np.ndarray,
    y_test_preds_lr: np.ndarray,
    y_test_preds_rf: np.ndarray,
) -> None:
    """Produces classification report for training and testing results.

    Additionally, it stores report as image in the images folder.

    Args:
        model_name: model name used for report.
        y_train: training response values.
        y_test:  test response values.
        y_train_preds_lr: training predictions from logistic regression.
        y_train_preds_rf: training predictions from random forest.
        y_test_preds_lr: test predictions from logistic regression.
        y_test_preds_rf: test predictions from random forest.

    Returns:
        None.
    """
    if model_name == "LogisticRegression":
        y_train_preds = y_train_preds_lr
        y_test_preds = y_test_preds_lr
    elif model_name == "RandomForestClassifier":
        y_train_preds = y_train_preds_rf
        y_test_preds = y_test_preds_rf
    else:
        logging.error(
            """
            Model name is not part of the classification_report_image function. Please
            update the function accordingly.
            """
        )

    plt.figure()
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str(f"{model_name} Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        str(f"{model_name} Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(image_path / f"results/{model_name}_classification_report.png")


def feature_importance_plot(model, model_name: str, X_data: pd.DataFrame) -> None:
    """
    Creates and stores the feature importances in pth.

    Args:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values

    Returns:
        None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(image_path / f"results/{model_name}_feature_importance.png")


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Train and store model results: images + scores, and store models.

    Args:
        X_train: X training data.
        X_test: X testing data.
        y_train: y training data.
        y_test: y testing data.

    Returns:
        None
    """
    logging.info("Starting to train models")
    # Train models and predict
    rfc = RandomForestClassifier(random_state=42, max_features="sqrt")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    logging.info("Successfully trained models")

    # Save models
    logging.info("Starting to save models")
    joblib.dump(cv_rfc.best_estimator_, model_path / "rfc_model.pkl")
    joblib.dump(lrc, model_path / "logistic_model.pkl")
    logging.info("Successfully saved models")

    logging.info("Starting to create images and reports")

    # ROC curves
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(image_path / "results/roc_curves.png")

    # Classification report
    for model in [lrc, cv_rfc]:
        model_name = (
            type(model).__name__
            if type(model).__name__ == "LogisticRegression"
            else type(model.estimator).__name__
        )

        classification_report_image(
            model_name,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
        )

    # Importance plot for RandomForestClassifier
    feature_importance_plot(cv_rfc, model_name, X_test)
    logging.info("Successfully created images and reports")


if __name__ == "__main__":
    df = import_data(data_path / "bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
