from diabetse.constants import *
from diabetse.config.configuration import *
import os ,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from diabetse.logger import logging
from diabetse.exception import CustomException

@dataclass
class DataInjectionConfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str=RAW_FILE_PATH

class DataInjection:
    def __init__(self):
        self.data_injection_config=DataInjectionConfig()

    def initiate_injection(self):
        try:
            logging.info("DATASET READING")
            df=pd.read_csv(DATASET_PATH)
            os.makedirs(os.path.dirname(self.data_injection_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_injection_config.raw_data_path,index=False)

            logging.info("DATASET SPLITING")
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_injection_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_injection_config.train_data_path,header=True,index=False)

            os.makedirs(os.path.dirname(self.data_injection_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.data_injection_config.test_data_path,header=True,index=False)

            logging.info("TRAIN SET,TEST SET RETURN")
            return(
                self.data_injection_config.train_data_path,
                self.data_injection_config.test_data_path
            )
        


        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataInjection()
    train_data,test_data=obj.initiate_injection()
