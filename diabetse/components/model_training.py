import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

from diabetse.constants import *
from diabetse.config.configuration import *
from diabetse.logger import logging
from diabetse.exception import CustomException
from diabetse.utils import save_obj

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/Tarun-898/Diabetes_prediction.mlflow")
import dagshub
dagshub.init(repo_owner='Tarun-898', repo_name='Diabetes_prediction', mlflow=True)




@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = MODEL_FILE_PATH


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_path, test_path):

        try:


            train_arr = pd.read_csv(
                train_path,
                header=None
            ).values

            test_arr = pd.read_csv(
                test_path,
                header=None
            ).values


            # --------------------------------
            # Split X and y
            # --------------------------------

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]


                # Encode target labels
            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)           


            logging.info(
                "Train and test data loaded successfully"
            )


            # --------------------------------
            # Models
            # --------------------------------

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=2000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        random_state=42,
                        n_jobs=-1
                    ),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        random_state=42
                    ),

                "XGBoost":
                    XGBClassifier(
                        random_state=42,
                        eval_metric="mlogloss"
                    )
            }


            # --------------------------------
            # Hyperparameters
            # --------------------------------

            params = {

                "Logistic Regression": {

                    "C": [0.01, 0.1, 1, 10, 100],

                    "solver": [
                        "lbfgs",
                        "liblinear"
                    ]
                },


                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy"
                    ],

                    "max_depth": [
                        3,
                        5,
                        10,
                        20,
                        None
                    ],

                    "min_samples_split": [
                        2,
                        5,
                        10
                    ]
                },


                "Random Forest": {

                    "n_estimators": [
                        100,
                        200,
                        300
                    ],

                    "max_depth": [
                        5,
                        10,
                        20,
                        None
                    ],

                    "min_samples_split": [
                        2,
                        5,
                        10
                    ]
                },


                "Gradient Boosting": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "max_depth": [
                        2,
                        3,
                        5
                    ]
                },


                "XGBoost": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "max_depth": [
                        3,
                        5,
                        7
                    ]
                }
            }


            # --------------------------------
            # Model evaluation
            # --------------------------------

            model_report = {}

            best_models = {}

            best_params = {}


            for model_name, model in models.items():

                logging.info(
                    f"Hyperparameter tuning started for {model_name}"
                )


                random_search = RandomizedSearchCV(

                    estimator=model,

                    param_distributions=params[model_name],

                    n_iter=10,

                    cv=5,

                    scoring="f1_weighted",

                    random_state=42,

                    n_jobs=-1
                )


                random_search.fit(
                    X_train,
                    y_train
                )


                best_model = (
                    random_search.best_estimator_
                )


                best_models[model_name] = best_model


                best_params[model_name] = (
                    random_search.best_params_
                )


                # Prediction

                y_pred = best_model.predict(
                    X_test
                )


                # F1 Score

                score = f1_score(
                    y_test,
                    y_pred,
                    average="weighted"
                )


                model_report[model_name] = score


                logging.info(
                    f"{model_name} F1 Score: {score}"
                )


                logging.info(
                    f"Best Parameters: "
                    f"{random_search.best_params_}"
                )


                print(
                    f"{model_name}: {score:.4f}"
                )


            # --------------------------------
            # Select best model
            # --------------------------------

            best_model_name = max(
                model_report,
                key=model_report.get
            )


            best_model_score = (
                model_report[best_model_name]
            )


            best_model = (
                best_models[best_model_name]
            )


            logging.info(
                f"Best Model: {best_model_name}"
            )

            logging.info(
                f"Best F1 Score: {best_model_score}"
            )


            print("\n==============================")

            print(
                f"Best Model: {best_model_name}"
            )

            print(
                f"Best F1 Score: "
                f"{best_model_score:.4f}"
            )

            print("==============================")


            # --------------------------------
            # Classification report
            # --------------------------------

            y_pred = best_model.predict(
                X_test
            )


            print("\nClassification Report:\n")

            print(
                classification_report(
                    y_test,
                    y_pred
                )
            )


            # --------------------------------
            # Save best model
            # --------------------------------

            save_obj(

                file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),

                obj=best_model
            )




            logging.info(
                "Best model saved successfully"
            )

            with mlflow.start_run(run_name=best_model_name):
                
                logging.info("mlflow start")
                mlflow.set_tag("developer","Tarun")
                mlflow.set_tag("project","diabetse prediction")

                y_pred=best_model.predict(X_test)
                score = f1_score(
                                    y_test,
                                    y_pred,
                                    average="weighted"
                                )

                mlflow.log_param("Model",best_model_name)
                mlflow.log_params(best_model.get_params())

                mlflow.log_metric("f1 score",score)

                logging.info("metrics logged successfully")

                mlflow.sklearn.log_model(sk_model=best_model,artifact_path="model")
                logging.info("model logged successfully")

                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)
                logging.info("Artifact Logged Successfully")


            return best_model_score


        except Exception as e:

            logging.exception(e)

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    try:

        train_path = TRANSFORM_TRAIN_PATH

        test_path = TRANSFORM_TEST_PATH


        obj = ModelTrainer()


        obj.initiate_model_training(
            train_path,
            test_path
        )


    except Exception as e:

        print(e)