from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.config import *

application = Flask(__name__)

app = application 

## Route for a home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:

        accountNumber = request.form.get('accountNumber')
        customerId = request.form.get('customerId')
        creditLimit = request.form.get('creditLimit')
        availableMoney = request.form.get('availableMoney')
        transactionDateTime = request.form.get('transactionDateTime')
        transactionAmount = request.form.get('transactionAmount')
        merchantName = request.form.get('merchantName')
        acqCountry = request.form.get('acqCountry')
        merchantCountryCode = request.form.get('merchantCountryCode')
        posEntryMode = request.form.get('posEntryMode')
        posConditionCode = request.form.get('posConditionCode')
        merchantCategoryCode = request.form.get('merchantCategoryCode')
        currentExpDate = request.form.get('currentExpDate')
        accountOpenDate = request.form.get('accountOpenDate')
        dateOfLastAddressChange = request.form.get('dateOfLastAddressChange')
        cardCVV = request.form.get('cardCVV')
        enteredCVV = request.form.get('enteredCVV')
        cardLast4Digits = request.form.get('cardLast4Digits')
        transactionType = request.form.get('transactionType')
        echoBuffer = request.form.get('echoBuffer')
        currentBalance = request.form.get('currentBalance')
        merchantCity = request.form.get('merchantCity')
        merchantState = request.form.get('merchantState')
        merchantZip = request.form.get('merchantZip')
        cardPresent = True if request.form.get('cardPresent')=="on" else False
        posOnPremises = request.form.get('posOnPremises')
        recurringAuthInd = request.form.get('recurringAuthInd')
        expirationDateKeyInMatch = True if request.form.get('expirationDateKeyInMatch')== "on" else False

        data = CustomData(

            accountNumber=accountNumber,
            customerId=customerId,
            creditLimit=creditLimit,
            availableMoney=availableMoney,
            transactionDateTime=transactionDateTime,
            transactionAmount=transactionAmount,
            merchantName=merchantName,
            acqCountry=acqCountry,
            merchantCountryCode=merchantCountryCode,
            posEntryMode=posEntryMode,
            posConditionCode=posConditionCode,
            merchantCategoryCode=merchantCategoryCode,
            currentExpDate=currentExpDate,
            accountOpenDate=accountOpenDate,
            dateOfLastAddressChange=dateOfLastAddressChange,
            cardCVV=cardCVV,
            enteredCVV=enteredCVV,
            cardLast4Digits=cardLast4Digits,
            transactionType=transactionType,
            echoBuffer=echoBuffer,
            currentBalance=currentBalance,
            merchantCity=merchantCity,
            merchantState=merchantState,
            merchantZip=merchantZip,
            cardPresent=cardPresent,
            posOnPremises=posOnPremises,
            recurringAuthInd=recurringAuthInd,
            expirationDateKeyInMatch=expirationDateKeyInMatch
        )

        pred_df = data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results= predict_pipeline.predict(pred_df)
        print("After Prediction")
        print(results)
        return render_template("home.html", results="Fraud" if results[0]==1 else "Regular Transaction")
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)



