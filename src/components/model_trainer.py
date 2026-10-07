import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


class ModelTrainer:

    def initiate_model_training(self):

        try:
            logging.info("Starting model training pipeline")

            train_path, test_path = DataIngestion().initiate_data_ingestion()

            X_train, y_train, X_test, y_test = (
                DataTransformation().initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            # Convert target from Yes/No to 1/0
            y_train = y_train.map({"No": 0, "Yes": 1})
            y_test = y_test.map({"No": 0, "Yes": 1})

            candidates = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000
                ),

                "RandomForest": RandomForestClassifier(
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42
                )
            }

            scores = {}

            print("\nModel Comparison")
            print("----------------")

            for name, model in candidates.items():

                logging.info(
                    f"Evaluating {name} using 5-fold cross-validation"
                )

                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="roc_auc"
                )

                mean_score = cv_scores.mean()

                scores[name] = mean_score

                print(
                    f"{name}: {mean_score:.4f}"
                )

                logging.info(
                    f"{name}: mean CV AUC = {mean_score:.4f}"
                )

            best_name = max(scores, key=scores.get)

            print(
                f"\nBest model: {best_name}"
            )

            print(
                f"Best CV AUC: {scores[best_name]:.4f}"
            )

            logging.info(
                f"Best model: {best_name}"
            )

            return (
                candidates,
                scores,
                best_name,
                X_train,
                y_train,
                X_test,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)