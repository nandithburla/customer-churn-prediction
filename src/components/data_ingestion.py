import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

            # Fix TotalCharges data quality issue found in Unit 3
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            )

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(
                self.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Splitting data into train and test sets")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Churn"]
            )

            train_set.to_csv(
                self.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)