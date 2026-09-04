import os, sys
import pandas as pd
import numpy as np
from diabetse.constants import *
from diabetse.config.configuration import *
from dataclasses import dataclass
from diabetse.logger import logging
from diabetse.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from diabetse.utils import save_obj
from diabetse.components.data_injection import *

@dataclass
class DataTransformationConfig():
    process_obj_file:str=PREPROCESSING_FILE
    transformed_train_file_path: str = TRANSFORM_TRAIN_PATH 
    transformed_test_file_path: str = TRANSFORM_TEST_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("*******PIPELINE SETUP STARTED**************")
            numerical_columns = [ "age", "bmi", 
                                 "hours_sleep_per_night", "stress_level", 
                                 "fasting_blood_sugar", "hba1c_level",
                                 "blood_pressure_systolic", "blood_pressure_diastolic",
                                   "waist_circumference_cm" ]

        
            categorical_columns = [ "gender", "city", "family_history_diabetes", 
                                   "diet_type", "smoking_status", "alcohol_consumption","physical_activity_level","income_bracket"]


            numerical_pipeline=Pipeline(steps=[("impute",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
            categorical_pipeline = Pipeline( steps=[ ( "imputer", SimpleImputer(strategy="most_frequent") ), ( "one_hot_encoder", OneHotEncoder( handle_unknown="ignore", sparse_output=False ) ) ] )

            preprocessor = ColumnTransformer( transformers=[ ( "num_pipeline", numerical_pipeline, numerical_columns ), 
                                                            ( "cat_pipeline", categorical_pipeline, categorical_columns ) 
                                                            ] )

            logging.info("*************PIPELINE STEUP DONE*************")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info("*************DATA TRANSFORMATION INITIATE*************")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            target_column_name="diabetes_risk"

            input_feature_train_df = train_df.drop( columns=[target_column_name] ) 
            target_feature_train_df = train_df[ target_column_name ] 


            input_feature_test_df = test_df.drop( columns=[target_column_name] ) 
            target_feature_test_df = test_df[ target_column_name ]

            processing_obj=self.get_data_transformation_obj()

            input_feature_train_arr=processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=processing_obj.transform(input_feature_test_df)

            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df) ] 
            test_arr = np.c_[ input_feature_test_arr, np.array(target_feature_test_df) ]

            df_train=pd.DataFrame(train_arr)
            df_test=pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path),exist_ok=True)
            os.makedirs( os.path.dirname( self.data_transformation_config .transformed_test_file_path ), exist_ok=True )

            logging.info("*************SAVING FILES*************")

            
            df_train.to_csv(self.data_transformation_config.transformed_train_file_path,index=False, header=False)
            df_test.to_csv( self.data_transformation_config .transformed_test_file_path, index=False, header=False )

            save_obj(file_path=self.data_transformation_config.process_obj_file,obj=processing_obj)

            logging.info("*************DATA TRANSFORMATION SUCCESSFULL*************")
            
            return ( self.data_transformation_config .transformed_train_file_path,
                     self.data_transformation_config .transformed_test_file_path, 
                     self.data_transformation_config .process_obj_file )
        
        except Exception as e:
             raise CustomException (e,sys)


if __name__ == "__main__": 
    try:  
        # obj=DataInjection()
        # train_data,test_data=obj.initiate_injection()

        train_file_path = TRAIN_FILE_PATH 
        test_file_path = TEST_FILE_PATH 
        obj = DataTransformation() 
        train_arr, test_arr, processor_path = ( 
            obj.initiate_data_transformation( train_file_path, test_file_path ) )
        print("Train transformed file:", train_arr) 
        print("Test transformed file:", test_arr) 
        print("Processor file:", processor_path) 
    except Exception as e: print(e)

