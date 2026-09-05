# import sys
# import pandas as pd

# from diabetse.config.configuration import (
#     PREPROCESSING_FILE,
#     MODEL_FILE_PATH
# )
# from diabetse.utils import load_obj
# from diabetse.exception import CustomException
# from diabetse.logger import logging


# class PredictionPipeline:

#     def __init__(self):
#         self.preprocessor_path = PREPROCESSING_FILE
#         self.model_path = MODEL_FILE_PATH

#     def predict(self, features):

#         try:
#             logging.info("Prediction pipeline started")

#             # Load preprocessing object
#             preprocessor = load_obj(
#                 file_path=self.preprocessor_path
#             )

#             # Load trained model
#             model = load_obj(
#                 file_path=self.model_path
#             )

#             # Convert user input into DataFrame
#             input_df = pd.DataFrame([features])

#             logging.info("Applying preprocessing")

#             # Apply the SAME preprocessing used during training
#             transformed_data = preprocessor.transform(input_df)

#             logging.info("Making prediction")

#             prediction = model.predict(transformed_data)

#             # Probability if model supports it
#             probability = None

#             if hasattr(model, "predict_proba"):
#                 probabilities = model.predict_proba(transformed_data)
#                 probability = float(max(probabilities[0])) * 100

#             logging.info("Prediction completed successfully")

#             return prediction[0], probability

#         except Exception as e:
#             logging.exception("Prediction failed")
#             raise CustomException(e, sys)