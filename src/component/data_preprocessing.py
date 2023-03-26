from src.logger import logging
from dataclasses import dataclass
import json
import os
import sys
import re
@dataclass

class DataPreprocessingConfig:
      nlp_mapping = json.load(open(os.path.join(os.getcwd(),'src','component','nlp_mapping.json'),'r'))

class DataPreprocessing:
      
    def __init__(self):
        self.data_preprocessing_config= DataPreprocessingConfig()

    def Accent_Normalization(self,text):
        import unidecode
        
        # remove ascents
        outputString = unidecode.unidecode(text)   
        
        return outputString  
       
    def Replace_Digit(self,text):
        return re.sub(r'\d', '', text).strip()
    
    def LowerCase(self,text):
        return text.lower()

    def StringTrimming(self,text):
        return text.strip()

    def ReomveWhiteSpace(self,text):
        return " ".join([words for words in text.split(" ") if words!=''])

    def RemoveSingleCharacter(self,text):
        return " ".join([words for words in text.split(" ") if not len(words)<=1])
    
    def Remove_Punctuation(self,text):
        
        text1 = self.LowerCase(text)
        text1 = self.Accent_Normalization(text1)
        text1 = self.Replace_Digit(text1)
        
        text1 = self.StringTrimming(text1)
        text1 = self.ReomveWhiteSpace(text1)
        text1 = self.RemoveSingleCharacter(text1)
        
        bag_of_symbols = [symbols for symbols in self.data_preprocessing_config.nlp_mapping['punctuation']]
        strings = "".join([words for words in text1 if words not in bag_of_symbols])
        return strings


    def Stemming(self,text):
        """
        It is also known as the text standardization step where the words are stemmed or diminished 
        to their root/base form.  For example, words like ‘programmer’, ‘programming, ‘program’
        will be stemmed to ‘program’.
        But the disadvantage of stemming is that it stems the words such that its root form loses
        the meaning or it is not diminished to a proper English word. We will see this in the steps done below.
        """    
        #importing the Stemming function from nltk library
        from nltk.stem.porter import PorterStemmer
        
        #defining the object for stemming
        porter_stemmer = PorterStemmer()
        
        #defining a function for stemming
        stem_text = ' '.join([porter_stemmer.stem(word) for word in text.split(" ")])

        return stem_text   
    
    def Lemmatizer(self,text):
        """
        It stems the word but makes sure that it does not lose its meaning.  
        Lemmatization has a pre-defined dictionary that stores the context of words and checks 
        the word in the dictionary while diminishing.

        The difference between Stemming and Lemmatization can be understood with the example provided below.

        Original Word	After Stemming	After Lemmatization
        goose	goos	goose
        geese	gees	goose
        """
        
        from nltk.stem import WordNetLemmatizer
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        
        #defining the object for Lemmatization
        wordnet_lemmatizer = WordNetLemmatizer()
        # self.__text = text
        lemm_text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split(" ")])
        #print(lemm_text)
        return lemm_text
    
    def Preprocessing(self,text):
        import warnings
        warnings.filterwarnings("ignore")
        #self.text = text
        text1 = self.Remove_Punctuation(text)
        text1 = self.Stemming(text1)
        text1 = self.Lemmatizer(text1)

        return text1

# if __name__=="__main__":

#     obj = DataPreprocessing()
#     print(obj.Preprocessing('eating  @1258605 +125       fish  @1258605 +125  like it stävänger'))