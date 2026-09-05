import sys
import pandas as pd

from diabetse.config.configuration import PREPROCESSING_FILE, MODEL_FILE_PATH
from diabetse.utils import load_obj
from diabetse.exception import CustomException
from diabetse.logger import logging

# Current trained dataset label mapping.
TARGET_LABELS = {
    0: "High",
    1: "Low",
    2: "Moderate",
}

class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = PREPROCESSING_FILE
        self.model_path = MODEL_FILE_PATH

    def predict(self, features):
        try:
            logging.info("Prediction pipeline started")

            preprocessor = load_obj(self.preprocessor_path)
            model = load_obj(self.model_path)

            input_df = pd.DataFrame([features])
            transformed_data = preprocessor.transform(input_df)

            prediction_encoded = int(model.predict(transformed_data)[0])
            prediction_label = TARGET_LABELS.get(
                prediction_encoded, str(prediction_encoded)
            )

            probabilities = {}
            confidence = None

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_data)[0]
                confidence = float(max(probs)) * 100

                for encoded_class, probability in zip(model.classes_, probs):
                    encoded_class = int(encoded_class)
                    probabilities[
                        TARGET_LABELS.get(encoded_class, str(encoded_class))
                    ] = float(probability) * 100

            return {
                "label": prediction_label,
                "encoded": prediction_encoded,
                "confidence": confidence,
                "probabilities": probabilities,
            }

        except Exception as e:
            logging.exception("Prediction failed")
            raise CustomException(e, sys)
