import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import *
from src.config import *

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Split train and test datasets")

            X_train, y_train, X_test, y_test = train_array[FINAL_FEATURES], train_array[TARGET], test_array[FINAL_FEATURES], test_array[TARGET]

            models = [
                ('Logistic Regression', LogisticRegression()),
                ('Decision Tree', DecisionTreeClassifier()),
                ('Random Forest', RandomForestClassifier()),
                ('Gradient Boosting', GradientBoostingClassifier()),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),  # To suppress a warning related to a deprecated feature in XGBoost
                ('LightGBM', LGBMClassifier())
            ]
            
            best_model_name, best_model_score, best_model = evaluate_models(X_train, y_train, X_test, y_test, models)

            

            if best_model_score<0.5:
                raise CustomException("No best model found")
            else:
                print("Best model score:", best_model_score)
                print("Best model:", best_model_name)

                logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            logging.error(f"Data transformation failed with error: {str(e)}")
            raise CustomException(str(e), sys.exc_info())