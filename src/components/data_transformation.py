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
from src.utils import save_object,data_cleaning, feature_engineering
import pickle

import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "processor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    # def get_data_transformation_object(self):
    #     setup_logging
    #     try:
    #         logging.info("Initiating data transformation")

            
    #         numeric_features = ['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present', 
    #                                 'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio',
    #                                   'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 'balance_ratio',
    #                                     'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 'transaction_hour', 
    #                                     'transaction_day_of_week', 'account_months', 'trans_gap_ratio_cus', 
    #                                     'trans_gap_ratio_card']
    #         categorical_features = ['is_acq_merchant_country_equal', 'is_correct_CVV']


    #         numeric_transformer = Pipeline(
    #             steps=[
    #                 ("imputer", SimpleImputer(strategy="median")),
    #                 ("scaler", StandardScaler())
    #             ]
    #         )

    #         categorical_transformer = Pipeline(
    #             steps=[
    #                 ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    #                 ("onehot", OneHotEncoder(handle_unknown="ignore"))
    #             ]
    #         )

    #         preprocessor = ColumnTransformer(
    #             transformers=[
    #                 ("num", numeric_transformer, numeric_features),
    #                 ("cat", categorical_transformer, categorical_features)
    #             ]
    #         )

    #         logging.info("Data transformation successful")

    #         return preprocessor

    #     except Exception as e:
    #         logging.error(CustomException(str(e), sys.exc_info()))
    #         sys.exit(1)
    

    def initiate_date_transformation(self, train_path, test_path):

        try:
            logging.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Data cleaning
            train_df = data_cleaning(train_df)
            test_df = data_cleaning(test_df)

            # Feature engineering
            train_df = feature_engineering(train_df)
            test_df = feature_engineering(test_df)

            # preprocessor = self.get_data_transformation_object()

            # target_column_name = "isFraud"
            # numeric_features = ['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present',
            #                      'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio',
            #                        'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 
            #                        'balance_ratio', 'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 
            #                        'transaction_hour', 'transaction_day_of_week', 'account_months', 
            #                        'trans_gap_ratio_cus', 'trans_gap_ratio_card']

            # input_feature_train_df = train_df.drop(target_column_name, axis=1)
            # target_feature_train_df = train_df[target_column_name]

            # input_feature_test_df = test_df.drop(target_column_name, axis=1)
            # target_feature_test_df = test_df[target_column_name]

            # def save_object(file_path, obj):
            #     with open(file_path, 'wb') as f:
            #         pickle.dump(obj, f)

            # logging.info("Fitting preprocessor")

            # input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data transformation successful")

            # save_object(
            #     file_path=self.transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessor
            # )

            return (
                train_df,
                test_df,
                # self.get_data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.error(f"Data transformation failed with error: {str(e)}")
            raise CustomException(str(e), sys.exc_info())
