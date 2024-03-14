import sys
import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import setup_logging
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "processor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        setup_logging
        try:
            logging.info("Initiating data transformation")
            numeric_features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            categorical_features = ["type"]

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features)
                ]
            )

            logging.info("Data transformation successful")

            return preprocessor

        except Exception as e:
            logging.error(CustomException(str(e), sys.exc_info()))
            sys.exit(1)
    

    def initiate_date_transformation(self, train_path, test_path):

        try:
            logging.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.get_data_transformation_object()

            target_column_name = "isFraud"
            numeric_features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            def save_object(file_path, obj):
                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)

            logging.info("Fitting preprocessor")

            input_feature_train_df = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            logging.info("Data transformation successful")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.get_data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.error(f"Data transformation failed with error: {str(e)}")
            raise CustomException(str(e), sys.exc_info())
