import os
import sys
import logging
import pandas as pd
import json

from ..exception import CustomException
from src.logger import setup_logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts",  "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")
    # sample_data_path: str=os.path.join("artifacts", "sample.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            file_path = '/Users/shouzhenghuang/fraud_dection/data/transactions.txt'

            def data_loading(path):
                # Open the file
                with open(file_path,'r') as file:
                    data = file.read()
                #Split it to list, use strip to drop '' in the end
                data_json = data.strip().split('\n')
                #Convert to dictionary
                data_dictionary = [json.loads(record) for record in data_json]
                #Replace `''` as None 
                data_dictionary = [{key: value if value != '' else None for key, value in record.items()} for record in data_dictionary]
                #Convert to dataframe
                df = pd.DataFrame(data_dictionary)

                return df

            df = data_loading(file_path)
            # df= pd.read_csv("/Users/shouzhenghuang/fraud_dection/data/transactions.txt")
            logging.info("Data ingestion successful")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data split successful")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            
            )


        except Exception as e:
            logging.error(f"Data ingestion failed with error: {str(e)}")
            raise CustomException(str(e), sys.exc_info())
        
if __name__ == "__main__":
    setup_logging()
    data_ingestion = DataIngestion()
    train_date, test_data =data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_date_transformation(train_date, test_data)

    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)
