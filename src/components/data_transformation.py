import pandas as pd
from sklearn.model_selection import train_test_split


def transform_data(df):
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Remove customer ID
    df = df.drop("customerID", axis=1)

    # Convert categorical columns to numbers
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df):
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )