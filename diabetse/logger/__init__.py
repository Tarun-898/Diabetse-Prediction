import logging
import os, sys
from datetime import datetime

LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_TIME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
NEW_PATH = f"log_{CURRENT_TIME}.log"

FILE_PATH = os.path.join(LOG_DIR, NEW_PATH)

logging.basicConfig(
    filename=FILE_PATH,
    filemode="w",
    format="[%(asctime)s]%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
)
