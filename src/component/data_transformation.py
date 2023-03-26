import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logging
from src.exception import CustomException

class DataTansformation:
      def __init__(self):
           # self.dataset = dataset
           pass
            
      def initiate_data_transformation(self,dataset):
            try:
                logging.info("Count Vectorizer used for performing Transformation") 
                self.dataset = dataset 
                self.countvector = CountVectorizer() #analyzer='word',max_features=5000)
                self.countvector.fit(self.dataset)

                self.__update_data = self.countvector.transform(self.dataset)

                return pd.DataFrame(self.__update_data.toarray(),
                                    columns=self.countvector.get_feature_names_out())
      
            except Exception as e:
                raise CustomException(e,sys)