import os
import sys
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer(
        self,
        numeric_cols,
        categorical_cols
    ):
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols)
                ]
            )

            logging.info("Preprocessing pipeline created")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "Churn"
            drop_cols = [target_col, "customerID"]

            numeric_cols = train_df.drop(
                columns=drop_cols
            ).select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_cols = train_df.drop(
                columns=drop_cols
            ).select_dtypes(
                include=["object", "str"]
            ).columns.tolist()

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(
                f"Categorical columns: {categorical_cols}"
            )

            preprocessor = self.get_data_transformer(
                numeric_cols,
                categorical_cols
            )

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target_col]

            # Learn preprocessing rules from training data
            X_train_arr = preprocessor.fit_transform(X_train)

            # Apply the learned rules to test data
            X_test_arr = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(
                "Saved preprocessor to artifacts/preprocessor.pkl"
            )

            logging.info("Data transformation completed")

            return (
                X_train_arr,
                y_train,
                X_test_arr,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)