import os
import sys

import numpy as np
import pandas as pd

import dill
import logging

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)


    except Exception as e:
        raise CustomException(str(e), sys.exc_info())
    

def data_cleaning(df):

    try:
        logging.info("Data cleaning started")

        # Convert to datetime
        datetime_columns = ['transactionDateTime','currentExpDate','accountOpenDate','dateOfLastAddressChange']
        df[datetime_columns] = df[datetime_columns].apply(pd.to_datetime)
    
        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Drop missing values
        df = df.drop(columns=['echoBuffer', 'merchantCity','merchantState','merchantZip','posOnPremises','recurringAuthInd'])

        # Fill missing values
        df['transactionType'].fillna("PURCHASE", inplace=True) # Most frequent value

        cols_to_fill = ['acqCountry','merchantCountryCode', 'posEntryMode','posConditionCode'] # Fill with N/A
        for col in cols_to_fill:
            df[col].fillna("N/A", inplace=True)


        # merchant name cleaning
        df['merchant_name'] = df['merchantName'].apply(lambda x : x.split('#')[0].strip(" "))   

        # Apply the mapping to the posEntryMode column
        mapping_pos_entry = {"05": 1, "80": 2, "02":3, "90": 4, "09": 5, "N/A":6}
        df["pos_Entry_Mode"] = df["posEntryMode"].map(mapping_pos_entry)

        # Apply the mapping to the posEntryMode column
        mapping_pos_condition = {"08": 1, "01": 2, "99": 3, "N/A":4}
        df["pos_Condition_Code"] = df["posConditionCode"].map(mapping_pos_condition)

        # Apply the mapping to the posEntryMode column
        mapping_transaction = {"ADDRESS_VERIFICATION": 1, "PURCHASE": 2, "REVERSAL": 2}
        df["transaction_type"] = df["transactionType"].map(mapping_transaction)

        # Apply the mapping to the posEntryMode column
        mapping_card_present = {True: 1, False: 2}
        df["card_present"] = df["cardPresent"].map(mapping_card_present)

        # Drop columns
        df.drop(columns=['merchantName', 'posEntryMode', 'posConditionCode','transactionType' ], inplace=True)
     

        logging.info("Data cleaning successful")

        return df

    except Exception as e:
        logging.error(CustomException(str(e), sys.exc_info()))
        sys.exit(1)

