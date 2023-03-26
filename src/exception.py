import sys
from src.logger import logging

def error_message_details(error,error_details:sys):
    _,_,exc_td = error_details.exc_info()
    line_number = exc_td.tb_lineno
    filename    = exc_td.tb_frame.f_code.co_filename

    error_message = f"Exception Occure in Python scripts at filename:{filename},line Number:{line_number},error message will be:{str(error)}"

    return error_message

class CustomException(Exception):
      def __init__(self,error_message,error_details:sys):
          super().__init__(error_message) 
          self.error_message = error_message_details(error=error_message,
                                                     error_details=error_details)
          
      def __str__(self):
          return self.error_message
      
# if __name__=="__main__":
# Testing Function Output

    # try:
        ## initialize the amount variable
        # marks = 10000

        ## perform division with 0
        # a = marks / 0
        
    # except Exception as e:
    #     b1 = CustomException(e,sys)
    #     print(e,'\n',b1)
    #     logging.info(b1)