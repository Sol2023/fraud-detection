# End-to-End Fraud Detection Project

## Purpose

The purpose of this project is to develop a machine learning model to identify fraudulent transactions based on customers' credit card transaction records.

## Tech Stack 

- Program langauge: Python, HTML, CSS;
- Devops: Flask, Docker, AWS,
- Version Control: Github, DVC

## Main Steps

### Part One: Jupyter Notebook

1. **Data Ingestion:** Obtain and collect the necessary data sources, such as credit card transaction records, for analysis.

2. **Data Preprocessing:** Implement data preprocessing techniques such as data cleaning, missing value imputation, and scaling to prepare the data for analysis.

3. **Exploratory Data Analysis (EDA):** Perform exploratory data analysis to understand the structure and characteristics of the data. This step involves data visualization, summary statistics, and identifying patterns or anomalies in the data.

4. **Feature Engineering:** Preprocess the data and engineer relevant features that will be used to train the machine learning model. Feature engineering may include normalization, transformation, encoding categorical variables, and creating new features to enhance model performance.

5. **Model Evaluation:** Develop modules for evaluating model performance, including confusion matrices, ROC curves, and precision-recall curves.

6. **Model Interpretability:** Integrate modules for interpreting model predictions and understanding feature importance to gain insights into the factors contributing to fraudulent transactions.

### Part Two: Engineering

1. Structure shows as below:

![image](https://github.com/Sol2023/fraud-detection/assets/92194263/b1ca07a6-6d8d-44f5-960b-9f1b008ff61e)


## Flask API

Applied flask api with front end page, where it allows user to input the transaction information for test purpose. When click predict, it will return the result that the transaction is Fruad or Normal Transaction

![image](https://github.com/Sol2023/fraud-detection/assets/92194263/0f336f43-035e-4640-960d-80f1c0d76025)

![image](https://github.com/Sol2023/fraud-detection/assets/92194263/7a22036a-5a71-4951-bde2-176f4e4cce5b)

![image](https://github.com/Sol2023/fraud-detection/assets/92194263/131324a4-c4a6-4eca-b8c9-8ebba42b3846)



## Deployment in AWS Cloud

Since my AWS account is no long active. So I just show the steps to deploy it in AWS cloud services using AWS Beanstalk

![image](https://github.com/Sol2023/fraud-detection/assets/92194263/3ce7f9c5-2205-4df7-a08b-b42fd70dcc10)


#### Step One: 
create `.ebextension` fold, under which create a config file cale `python.config`, content shows as below: 
```
option_settings: 
  "aws:elasticbeanstalk: container:python": 
    WSGAPath: application:application
```

#### Step Two:
change the name of `app.py` to `application.py` and delete debug=True as below:
```
if __name__=="__main__":
    app.run(host="0.0.0.0")
```

#### Step Three:

Follow this step by step CI/CD setup [link](https://aws.plainenglish.io/setup-a-ci-cd-pipeline-using-aws-codepipeline-to-deploy-a-node-js-application-to-elastic-beanstalk-5c75fcaf72e0)
