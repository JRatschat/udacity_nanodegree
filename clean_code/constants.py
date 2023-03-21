"""
Constants for use in other files.

Author: Jonathan Ratschat
Date: 21.03.2022
"""

from pathlib import Path

# paths
root_path = Path.cwd() / "clean_code"
data_path = root_path / "data"
image_path = root_path / "images"
model_path = root_path / "models"
log_path = root_path / "logs"

# feature engineering
categorical_cols_lst = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

keep_cols_lst = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]

# training
param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt"],
    "max_depth": [4, 5, 100],
    "criterion": ["gini", "entropy"],
}

# testing
eda_imgs_ls = [
    "churn_histogram.png",
    "correlation_heatmap.png",
    "customer_age_histogram.png",
    "marital_status_value_counts_norm_bar.png",
    "total_trans_density.png",
]

train_imgs_ls = [
    "LogisticRegression_classification_report.png",
    "RandomForestClassifier_classification_report.png",
    "RandomForestClassifier_feature_importance.png",
    "roc_curves.png",
]

models_ls = [
    "logistic_model.pkl",
    "rfc_model.pkl",
]
