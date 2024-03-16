import os
import sys

import numpy as np
import pandas as pd
import json 

from sklearn.metrics import f1_score

import dill
import pickle
import logging

from src.exception import CustomException

import warnings
warnings.filterwarnings('ignore')

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


def feature_engineering(df):


    try:
        logging.info("Feature engineering started")

        # Merchant transaction outlier
        merchant_transaction_summary = []

        for merchant in df['merchant_name'].unique():
            sub_df = df[df['merchant_name']==merchant]
            trans = sub_df['transactionAmount'].describe().to_dict()
            merchant_transaction_summary.append([merchant, trans])

        merchant_transaction_summary = pd.DataFrame(merchant_transaction_summary, columns=['merchant_name', 'merchant_trans_summary'])

        df = pd.merge(df, merchant_transaction_summary, on='merchant_name', how='left')

        def classify_merchant_trans(transactionAmount, merchant_trans_summary):
            p_75 = merchant_trans_summary['75%']
            p_25 = merchant_trans_summary['25%']
            IQR= (p_75-p_25)*1.5
            if transactionAmount > p_75+IQR:
                return 3 # transaction amount too big
            elif transactionAmount< p_25 - IQR:
                return 1 # transaction amount too samll
            else:
                return 2 # transaction amount normal

        df['merchant_trans_outlier'] = df.apply(lambda x: classify_merchant_trans(x.transactionAmount, x.merchant_trans_summary), axis=1)

        logging.info("merchant_trans_outlier created")

        #Fraud ratio by merchant code

        merchant_category_safety = pd.pivot_table(data=df, index ='merchantCategoryCode',
                                          columns='isFraud',
                                          values = 'customerId',
                                          fill_value=0,
                                          aggfunc='count')
        merchant_category_safety['merchant_code_fraud_ratio'] = merchant_category_safety[True] / (merchant_category_safety[True]+merchant_category_safety[False])

        merchant_category_safety.reset_index(inplace=True)

        df = pd.merge(df, merchant_category_safety[['merchantCategoryCode', 'merchant_code_fraud_ratio']], on='merchantCategoryCode')

        logging.info("merchant_code_fraud_ratio created")

        #Fraud ratio by merchant name
        merchant_name_fraud = pd.pivot_table(data=df, index='merchant_name', columns='isFraud',
                                     values='customerId', aggfunc='count', fill_value=0).reset_index()
        merchant_name_fraud['merchant_name_fraud_ratio'] = merchant_name_fraud[True]/ (merchant_name_fraud[True]+ merchant_name_fraud[False])


        df = pd.merge(df, merchant_name_fraud[['merchant_name', 'merchant_name_fraud_ratio']], on ='merchant_name', how='left')

        logging.info("merchant_name_fraud_ratio created")

        # Transaction gap
        # Day gap between current datetime and last transaction datatime by customer
        df.sort_values(by=['customerId', 'transactionDateTime'], inplace=True)
        df['cus_trans_gap_by_second'] = df.groupby(['customerId'])['transactionDateTime'].diff()

        # Day gap between current datetime and last transaction datatime by customer, by card
        df.sort_values(by=['customerId', 'cardLast4Digits','transactionDateTime'], inplace=True)
        df['cus_card_trans_gap_by_second'] = df.groupby(['customerId','cardLast4Digits'])['transactionDateTime'].diff()

        def get_second(gap):
            try:
                res = gap.seconds
            except:
                res=-1
            return res 

        df['cus_trans_gap_by_second'] = df['cus_trans_gap_by_second'].apply(get_second)
        df['cus_card_trans_gap_by_second'] = df['cus_card_trans_gap_by_second'].apply(get_second)

        trans_freq_by_cus = df.groupby('customerId')['cus_trans_gap_by_second'].mean()
        trans_freq_by_cus = pd.DataFrame({'customerId': trans_freq_by_cus.index, 'mean_cus_trans_gap': trans_freq_by_cus.values})

        df = pd.merge(df, trans_freq_by_cus[['customerId', 'mean_cus_trans_gap']], on ='customerId', how='left')

        trans_freq_by_cus_by_car = df.groupby('customerId')['cus_card_trans_gap_by_second'].mean()
        trans_freq_by_cus_by_car = pd.DataFrame({'customerId': trans_freq_by_cus_by_car.index, 'mean_cus_card_trans_gap': trans_freq_by_cus_by_car.values})

        df = pd.merge(df, trans_freq_by_cus_by_car[['customerId', 'mean_cus_card_trans_gap']], on ='customerId', how='left')

        df['trans_gap_ratio_cus'] = df['cus_card_trans_gap_by_second'] / df['mean_cus_card_trans_gap']
        df['trans_gap_ratio_card'] = df['cus_trans_gap_by_second'] / df['mean_cus_card_trans_gap']

        df['trans_gap_ratio_cus'] = df['cus_card_trans_gap_by_second'] / df['mean_cus_card_trans_gap']
        df['trans_gap_ratio_card'] = df['cus_trans_gap_by_second'] / df['mean_cus_card_trans_gap']

        logging.info("transaction gap features created")

        #Credit ratio
        df['credit_used_ratio'] = df['transactionAmount'] / df['creditLimit']
        df['credit_left_ratio'] = df['availableMoney']/ df['creditLimit']
        df['balance_ratio'] = df['currentBalance'] / df['creditLimit']

        logging.info("credit ratio features created")

        # Transaction ratio by category and merchant
        customer_category_summary = pd.pivot_table(data = df[df['isFraud']==False],
                                           index = ['customerId', 'merchantCategoryCode'],
                                           values = 'transactionAmount',
                                           aggfunc=np.mean
                                           ).reset_index()

        customer_category_summary.rename(columns= {"transactionAmount": "cus_transaction_avg_amount_by_category"}, inplace=True)

        df = pd.merge(df, customer_category_summary, on=['customerId', 'merchantCategoryCode'], how='left')
        df['cus_trans_ratio_by_cat'] = df['transactionAmount'] / df['cus_transaction_avg_amount_by_category']

        customer_merchant_summary = pd.pivot_table(data = df[df['isFraud']==False],
                                           index = ['customerId', 'merchant_name'],
                                           values = 'transactionAmount',
                                           aggfunc=np.mean
                                           ).reset_index()

        customer_merchant_summary.rename(columns= {"transactionAmount": "cus_transaction_avg_amount_by_merchant"}, inplace=True)

        df = pd.merge(df, customer_merchant_summary, on=['customerId', 'merchant_name'], how='left')
        df['cus_trans_ratio_by_merchant'] = df['transactionAmount'] / df['cus_transaction_avg_amount_by_merchant']

        logging.info("transaction ratio by category and merchant features created")

        # Boolean Features
        #Combine columns cardCVV and enteredCVV as is_correct_CVV
        df['is_correct_CVV'] = df['cardCVV']==df['enteredCVV']

        df['is_acq_merchant_country_equal'] = df['acqCountry'] ==df['merchantCountryCode']

        logging.info("Boolean features created")

        # # Extract features from 'transactionDataTime'
        df['transaction_hour'] = df['transactionDateTime'].dt.hour
        df['transaction_day_of_week'] = df['transactionDateTime'].dt.dayofweek
        df['account_months'] = df['transactionDateTime'].dt.month - df['accountOpenDate'].dt.month

        logging.info("Time features created")
        
        features = [
            'is_acq_merchant_country_equal', 'pos_Entry_Mode', 'pos_Condition_Code',
            'transaction_type', 'card_present', 
            'merchant_trans_outlier', 'merchant_code_fraud_ratio',
            'merchant_name_fraud_ratio', 
            'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio',
            'balance_ratio',
            'cus_trans_ratio_by_cat', 
            'cus_trans_ratio_by_merchant', 'is_correct_CVV', 'transaction_hour',
            'transaction_day_of_week', 'account_months','trans_gap_ratio_cus','trans_gap_ratio_card']

        id = ['accountNumber','customerId']
        target = 'isFraud'

        df_cleaned = df[features+id+[target]]

        df_cleaned.fillna(-99, inplace=True)
        df_cleaned.replace(np.inf, 999, inplace=True)

        df_cleaned[['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present', 
                    'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio', 
                    'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 
                    'balance_ratio', 'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 
                    'transaction_hour', 'transaction_day_of_week', 'account_months', 'trans_gap_ratio_cus', 
                    'trans_gap_ratio_card']] = df_cleaned[['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present', 
                    'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio', 
                    'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 
                    'balance_ratio', 'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 
                    'transaction_hour', 'transaction_day_of_week', 'account_months', 'trans_gap_ratio_cus', 
                    'trans_gap_ratio_card']].astype("float")
        
        df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV', 'isFraud']] = df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV', 'isFraud']].astype("boolean")


        logging.info("Feature engineering successful")

        return df_cleaned


    except Exception as e:

        logging.error(CustomException(str(e), sys.exc_info()))
        sys.exit(1)

def feature_engineering_test(df):


    try:
        logging.info("Feature engineering started")

        # Merchant transaction outlier
        merchant_transaction_summary = pd.read_csv("artifacts/merchant_transaction_summary.csv")
        # merchant_transaction_summary = []

        # for merchant in df['merchant_name'].unique():
        #     sub_df = df[df['merchant_name']==merchant]
        #     trans = sub_df['transactionAmount'].describe().to_dict()
        #     merchant_transaction_summary.append([merchant, trans])

        # merchant_transaction_summary = pd.DataFrame(merchant_transaction_summary, columns=['merchant_name', 'merchant_trans_summary'])

        df = pd.merge(df, merchant_transaction_summary, on='merchant_name', how='left')

        def classify_merchant_trans(transactionAmount, merchant_trans_summary):
            merchant_trans_summary=json.loads(merchant_trans_summary.replace("'", '"'))
            p_75 = merchant_trans_summary['75%']
            p_25 = merchant_trans_summary['25%']
            IQR= (p_75-p_25)*1.5
            if float(transactionAmount) > p_75+IQR:
                return 3 # transaction amount too big
            elif float(transactionAmount)< p_25 - IQR:
                return 1 # transaction amount too samll
            else:
                return 2 # transaction amount normal

        df['merchant_trans_outlier'] = df.apply(lambda x: classify_merchant_trans(x.transactionAmount, x.merchant_trans_summary), axis=1)

        logging.info("merchant_trans_outlier created")

        #Fraud ratio by merchant code
        merchant_category_safety = pd.read_csv("artifacts/merchant_category_safety.csv")
        # merchant_category_safety = pd.pivot_table(data=df, index ='merchantCategoryCode',
        #                                   columns='isFraud',
        #                                   values = 'customerId',
        #                                   fill_value=0,
        #                                   aggfunc='count')
        # merchant_category_safety['merchant_code_fraud_ratio'] = merchant_category_safety[True] / (merchant_category_safety[True]+merchant_category_safety[False])

        # merchant_category_safety.reset_index(inplace=True)

        df = pd.merge(df, merchant_category_safety[['merchantCategoryCode', 'merchant_code_fraud_ratio']], on='merchantCategoryCode')

        logging.info("merchant_code_fraud_ratio created")

        #Fraud ratio by merchant name

        merchant_name_fraud = pd.read_csv("artifacts/merchant_name_fraud.csv")
        # merchant_name_fraud = pd.pivot_table(data=df, index='merchant_name', columns='isFraud',
        #                              values='customerId', aggfunc='count', fill_value=0).reset_index()
        # merchant_name_fraud['merchant_name_fraud_ratio'] = merchant_name_fraud[True]/ (merchant_name_fraud[True]+ merchant_name_fraud[False])


        df = pd.merge(df, merchant_name_fraud[['merchant_name', 'merchant_name_fraud_ratio']], on ='merchant_name', how='left')

        logging.info("merchant_name_fraud_ratio created")

        # Transaction gap
        # Day gap between current datetime and last transaction datatime by customer
        df.sort_values(by=['customerId', 'transactionDateTime'], inplace=True)
        df['cus_trans_gap_by_second'] = df.groupby(['customerId'])['transactionDateTime'].diff()

        # Day gap between current datetime and last transaction datatime by customer, by card
        df.sort_values(by=['customerId', 'cardLast4Digits','transactionDateTime'], inplace=True)
        df['cus_card_trans_gap_by_second'] = df.groupby(['customerId','cardLast4Digits'])['transactionDateTime'].diff()

        def get_second(gap):
            try:
                res = gap.seconds
            except:
                res=-1
            return res 

        df['cus_trans_gap_by_second'] = df['cus_trans_gap_by_second'].apply(get_second)
        df['cus_card_trans_gap_by_second'] = df['cus_card_trans_gap_by_second'].apply(get_second)

        trans_freq_by_cus = df.groupby('customerId')['cus_trans_gap_by_second'].mean()
        trans_freq_by_cus = pd.DataFrame({'customerId': trans_freq_by_cus.index, 'mean_cus_trans_gap': trans_freq_by_cus.values})

        df = pd.merge(df, trans_freq_by_cus[['customerId', 'mean_cus_trans_gap']], on ='customerId', how='left')

        trans_freq_by_cus_by_car = df.groupby('customerId')['cus_card_trans_gap_by_second'].mean()
        trans_freq_by_cus_by_car = pd.DataFrame({'customerId': trans_freq_by_cus_by_car.index, 'mean_cus_card_trans_gap': trans_freq_by_cus_by_car.values})

        df = pd.merge(df, trans_freq_by_cus_by_car[['customerId', 'mean_cus_card_trans_gap']], on ='customerId', how='left')

        df['trans_gap_ratio_cus'] = df['cus_card_trans_gap_by_second'] / df['mean_cus_card_trans_gap']
        df['trans_gap_ratio_card'] = df['cus_trans_gap_by_second'] / df['mean_cus_card_trans_gap']

        df['trans_gap_ratio_cus'] = df['cus_card_trans_gap_by_second'] / df['mean_cus_card_trans_gap']
        df['trans_gap_ratio_card'] = df['cus_trans_gap_by_second'] / df['mean_cus_card_trans_gap']

        logging.info("transaction gap features created")

        #Credit ratio
        df['credit_used_ratio'] = float(df['transactionAmount']) / float(df['creditLimit'])
        df['credit_left_ratio'] = float(df['availableMoney'])/ float(df['creditLimit'])
        df['balance_ratio'] = float(df['currentBalance']) / float(df['creditLimit'])

        logging.info("credit ratio features created")

        # Transaction ratio by category and merchant
        # customer_category_summary = pd.pivot_table(data = df[df['isFraud']==False],
        #                                    index = ['customerId', 'merchantCategoryCode'],
        #                                    values = 'transactionAmount',
        #                                    aggfunc=np.mean
        #                                    ).reset_index()

        # customer_category_summary.rename(columns= {"transactionAmount": "cus_transaction_avg_amount_by_category"}, inplace=True)
        customer_category_summary = pd.read_csv("artifacts/customer_category_summary.csv")

        customer_category_summary[['customerId', 'merchantCategoryCode']]=customer_category_summary[['customerId', 'merchantCategoryCode']].astype("category")
        df[['customerId', 'merchantCategoryCode']] =df[['customerId', 'merchantCategoryCode']].astype("category")

        df = pd.merge(df, customer_category_summary, on=['customerId', 'merchantCategoryCode'], how='left')
        df['cus_trans_ratio_by_cat'] = float(df['transactionAmount']) / float(df['cus_transaction_avg_amount_by_category'])

        # customer_merchant_summary = pd.pivot_table(data = df[df['isFraud']==False],
        #                                    index = ['customerId', 'merchant_name'],
        #                                    values = 'transactionAmount',
        #                                    aggfunc=np.mean
        #                                    ).reset_index()

        # customer_merchant_summary.rename(columns= {"transactionAmount": "cus_transaction_avg_amount_by_merchant"}, inplace=True)

        customer_merchant_summary = pd.read_csv("artifacts/customer_merchant_summary.csv")

        customer_merchant_summary[['customerId', 'merchant_name']]=customer_merchant_summary[['customerId', 'merchant_name']].astype("category")
        df[['customerId', 'merchant_name']] =df[['customerId', 'merchant_name']].astype("category")

        df = pd.merge(df, customer_merchant_summary, on=['customerId', 'merchant_name'], how='left')
        df['cus_trans_ratio_by_merchant'] = float(df['transactionAmount']) / float(df['cus_transaction_avg_amount_by_merchant'])

        logging.info("transaction ratio by category and merchant features created")

        # Boolean Features
        #Combine columns cardCVV and enteredCVV as is_correct_CVV
        df['is_correct_CVV'] = df['cardCVV']==df['enteredCVV']

        df['is_acq_merchant_country_equal'] = df['acqCountry'] ==df['merchantCountryCode']

        logging.info("Boolean features created")

        # # Extract features from 'transactionDataTime'
        df['transaction_hour'] = df['transactionDateTime'].dt.hour
        df['transaction_day_of_week'] = df['transactionDateTime'].dt.dayofweek
        df['account_months'] = df['transactionDateTime'].dt.month - df['accountOpenDate'].dt.month

        logging.info("Time features created")
        
        features = [
            'is_acq_merchant_country_equal', 'pos_Entry_Mode', 'pos_Condition_Code',
            'transaction_type', 'card_present', 
            'merchant_trans_outlier', 'merchant_code_fraud_ratio',
            'merchant_name_fraud_ratio', 
            'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio',
            'balance_ratio',
            'cus_trans_ratio_by_cat', 
            'cus_trans_ratio_by_merchant', 'is_correct_CVV', 'transaction_hour',
            'transaction_day_of_week', 'account_months','trans_gap_ratio_cus','trans_gap_ratio_card']

        id = ['accountNumber','customerId']
        target = 'isFraud'

        if 'isFraud' not in df.columns:
            # df['isFraud'] = False
            df_cleaned = df[features+id]
        else:
            df_cleaned = df[features+id+[target]]

        df_cleaned.fillna(-99, inplace=True)
        df_cleaned.replace(np.inf, 999, inplace=True)

        df_cleaned[['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present', 
                    'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio', 
                    'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 
                    'balance_ratio', 'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 
                    'transaction_hour', 'transaction_day_of_week', 'account_months', 'trans_gap_ratio_cus', 
                    'trans_gap_ratio_card']] = df_cleaned[['pos_Entry_Mode', 'pos_Condition_Code', 'transaction_type', 'card_present', 
                    'merchant_trans_outlier', 'merchant_code_fraud_ratio', 'merchant_name_fraud_ratio', 
                    'mean_cus_card_trans_gap', 'credit_used_ratio', 'credit_left_ratio', 
                    'balance_ratio', 'cus_trans_ratio_by_cat', 'cus_trans_ratio_by_merchant', 
                    'transaction_hour', 'transaction_day_of_week', 'account_months', 'trans_gap_ratio_cus', 
                    'trans_gap_ratio_card']].astype("float")
        
        if 'isFraud' not in df.columns:
            df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV']] = df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV']].astype("boolean")

        else:
            df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV', 'isFraud']] = df_cleaned[['is_acq_merchant_country_equal', 'is_correct_CVV', 'isFraud']].astype("boolean")


        logging.info("Feature engineering successful")

        return df_cleaned


    except Exception as e:

        logging.error(CustomException(str(e), sys.exc_info()))
        sys.exit(1)



def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:

        f1_scores = {}

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_scores[name] = f1_score(y_test, y_pred)

        # Print each model's F1 score
        f1_scores_df = pd.DataFrame(list(f1_scores.items()), columns=['Model', 'F1 Score'])
        
        print(f1_scores_df)

        # return f1_scores

        # Find the best model based on the F1 score
        best_model_name = max(f1_scores, key=f1_scores.get)
        best_model_score = f1_scores[best_model_name]
        best_model = None
        for name, model in models:
            if name == best_model_name:
                best_model = model
                break
        
        return (
            best_model_name,
            best_model_score,
            best_model
        )

    except Exception as e:
        logging.error(CustomException(str(e), sys.exc_info()))
        sys.exit(1)

def get_model_by_name(model_name, models):
    for name, model in models:
        if name == model_name:
            return model
    return None  # Return None if model_name is not found

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)