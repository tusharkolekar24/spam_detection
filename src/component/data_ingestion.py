import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.component.data_transformation import DataTansformation
from src.component.data_preprocessing import DataPreprocessing
from sklearn.model_selection import train_test_split
@dataclass
class DataIngestionConfig:
      source_file_path     :str = os.path.join(os.getcwd(),'dataset','source_file','spam_messages.csv')
      raw_data_store_path  :str = os.path.join(os.getcwd(),'artifacts','raw_data.csv')
      train_data_store_path:str = os.path.join(os.getcwd(),'artifacts','train_dataset.csv')
      test_data_store_path :str = os.path.join(os.getcwd(),'artifacts','test_dataset.csv')

class DataIngestion:
      def __init__(self):
           self.data_extraction_config = DataIngestionConfig()

      def initiate_data_ingestion(self):
           logging.info("Data Ingestion process Start")
           try:
               insert_dataset = pd.read_csv(self.data_extraction_config.source_file_path,
                                            encoding="ISO-8859-1")

               insert_dataset.columns = ['message','target']
               
               logging.info('Read the dataset as dataframe')

               self.process = DataPreprocessing()
               corpus = [self.process.Preprocessing(sentence) for sentence in insert_dataset['message']]

               self.transformation = DataTansformation()
               transforme_dataset = self.transformation.initiate_data_transformation(corpus)

               transforme_dataset['target'] = insert_dataset['target'].map({"ham":1,"spam":0})

               transforme_dataset.to_csv(self.data_extraction_config.raw_data_store_path,
                                         index=False,
                                         header=True)
               
               logging.info("Transformation Perform on Inserted Dataset")

               trainX, testX = train_test_split(transforme_dataset,
                                                  test_size=0.35,
                                                  random_state=42)
               
               trainX.to_csv(self.data_extraction_config.train_data_store_path,
                             index=False,header=True)
               
               testX.to_csv(self.data_extraction_config.test_data_store_path,
                            index=False,header=True)
               
               return (self.data_extraction_config.train_data_store_path,
                       self.data_extraction_config.test_data_store_path)
           
           except Exception as e:
               raise CustomException(e,sys)   

# if __name__=="__main__":
#      obj = DataIngestion()    
#      a,b = obj.initiate_data_ingestion()           
