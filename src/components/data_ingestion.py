import pandas as pd


def load_data():
    data_path = "data/raw/Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)

    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")

    return df