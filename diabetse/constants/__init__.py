import os, sys
from datetime import datetime


def get_current_time():
    return f"{ datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


CURRENT_TIME = get_current_time()

ROOT_DIR = os.getcwd()
DATA_DIR = "data_dir"
DATASET = "diabetes_risk.csv"

ARTIFACT_DIR = "artifact"

DATA_INGESTION = "data_ingestion"
RAW_DATA_DIR = "raw_data_dir"
INGESTED_DATA_DIR = "ingested_data_dir"

RAW_DATA = "raw.csv"
TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"

# data Transformation constants

DATA_TRANSFORMATION = "data_transformation"

PREPROCESS_DIR = "processor_dir"
PROCESS_DATA = "processor.pkl"

TRANSFORMATION_DIR = "transformation_dir"
TRANSFORM_TRAIN_DATA = "train.csv"
TRANSFORM_TEST_DATA = "test.csv"

# model training constants
MODEL_TRAINER_DIR = "model_trainer_dir"
MODEL_OBJECT = "model.pkl"
