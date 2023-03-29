from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import Save_Object,Evaluate_Models
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys
import os

@dataclass

class DataPredictionConfig:
      testset_file_path = os.path.join(os.getcwd(),"artifacts","test_dataset.csv")
      trainset_file_path = os.path.join(os.getcwd(),"artifacts","train_dataset.csv")
      file_path = os.path.join(os.getcwd(),"artifacts") 

class DataPredictionPipeline:
      def __init__(self):
            self.data_training_config = DataPredictionConfig()

      def initiate_data_testing_pipeline(self):
           logging.info("Data Testing Pipeline Initiated") 
           try:
                testset  =  pd.read_csv(self.data_training_config.testset_file_path)
                logging.info("Testing Data available in artifacts")

                trainset =  pd.read_csv(self.data_training_config.trainset_file_path)
                logging.info("Training Data available in artifacts")

                X_test, y_test  = testset.iloc[:,:-1], testset.iloc[:,-1]
                X_train,y_train = trainset.iloc[:,:-1],trainset.iloc[:,-1]

                logging.info("Dataset are split into Train Test")

                evaluate_model_performance:dict = Evaluate_Models(X_train,X_test,y_train,y_test)
                
                #print(evaluate_model_performance)
                model_performance_summary = pd.DataFrame(evaluate_model_performance)

                model_performance_summary.to_csv(os.path.join(os.getcwd(),"artifacts",'model_performance_summary.csv'),
                                                 index=False)
                
                return model_performance_summary

           except Exception as e:
                  
                  raise CustomException(e,sys)
           
if __name__=="__main__":
      obj = DataPredictionPipeline()
      r1 = obj.initiate_data_testing_pipeline()
