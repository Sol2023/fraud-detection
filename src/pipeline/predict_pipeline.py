import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import *
from src.logger import *


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        setup_logging()
        try:
            model_path=os.path.join("artifacts","model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            # preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            # data_scaled=preprocessor.transform(features)
            data_cleaned = data_cleaning(features)
            data_cleaned.to_csv("artifacts/sample_cleaned.csv",index=False)
            data_cleaned = feature_engineering(data_cleaned)
            
            preds=model.predict(data_cleaned)
            print(preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        accountNumber, 
        customerId,
        creditLimit, 
        availableMoney,
        transactionDateTime,
        transactionAmount, 
        merchantName,
        acqCountry,
        merchantCountryCode,
        posEntryMode,
        posConditionCode,
        merchantCategoryCode,
        currentExpDate, 
        accountOpenDate,
        dateOfLastAddressChange, 
        cardCVV, 
        enteredCVV, 
        cardLast4Digits,
        transactionType,
        echoBuffer, 
        currentBalance,
        merchantCity,
        merchantState, 
        merchantZip,
        cardPresent, 
        posOnPremises,
        recurringAuthInd,
        expirationDateKeyInMatch
        ):

        self.accountNumber = accountNumber
        self.customerId = customerId
        self.creditLimit = creditLimit
        self.availableMoney = availableMoney
        self.transactionDateTime = transactionDateTime
        self.transactionAmount = transactionAmount
        self.merchantName = merchantName
        self.acqCountry = acqCountry
        self.merchantCountryCode = merchantCountryCode
        self.posEntryMode = posEntryMode
        self.posConditionCode = posConditionCode
        self.merchantCategoryCode = merchantCategoryCode
        self.currentExpDate = currentExpDate
        self.accountOpenDate = accountOpenDate
        self.dateOfLastAddressChange = dateOfLastAddressChange
        self.cardCVV = cardCVV
        self.enteredCVV = enteredCVV
        self.cardLast4Digits = cardLast4Digits
        self.transactionType = transactionType
        self.echoBuffer = echoBuffer
        self.currentBalance = currentBalance
        self.merchantCity = merchantCity
        self.merchantState = merchantState
        self.merchantZip = merchantZip
        self.cardPresent = cardPresent
        self.posOnPremises = posOnPremises
        self.recurringAuthInd = recurringAuthInd
        self.expirationDateKeyInMatch = expirationDateKeyInMatch



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'accountNumber': [self.accountNumber],
                'customerId': [self.customerId],
                'creditLimit': [self.creditLimit],
                'availableMoney': [self.availableMoney],
                'transactionDateTime': [self.transactionDateTime],
                'transactionAmount': [self.transactionAmount],
                'merchantName': [self.merchantName],
                'acqCountry': [self.acqCountry],
                'merchantCountryCode': [self.merchantCountryCode],
                'posEntryMode': [self.posEntryMode],
                'posConditionCode': [self.posConditionCode],
                'merchantCategoryCode': [self.merchantCategoryCode],
                'currentExpDate': [self.currentExpDate],
                'accountOpenDate': [self.accountOpenDate],
                'dateOfLastAddressChange': [self.dateOfLastAddressChange],
                'cardCVV': [self.cardCVV],
                'enteredCVV': [self.enteredCVV],
                'cardLast4Digits': [self.cardLast4Digits],
                'transactionType': [self.transactionType],
                'echoBuffer': [self.echoBuffer],
                'currentBalance': [self.currentBalance],
                'merchantCity': [self.merchantCity],
                'merchantState': [self.merchantState],
                'merchantZip': [self.merchantZip],
                'cardPresent': [self.cardPresent],
                'posOnPremises': [self.posOnPremises],
                'recurringAuthInd': [self.recurringAuthInd],
                'expirationDateKeyInMatch': [self.expirationDateKeyInMatch]
            }

            df = pd.DataFrame(custom_data_input_dict)
            sample_data_path = os.path.join("artifacts", "sample.csv")
            df.to_csv(sample_data_path,index=False)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
