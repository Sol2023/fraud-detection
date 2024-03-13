import logging
import sys
import logging

from logger import setup_logging

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in file {file_name}, line {exc_tb.tb_lineno}: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    setup_logging()
    
    try:
        a = 1/0
    except Exception as e:
        logging.error(CustomException(str(e), sys.exc_info()))
        print(CustomException(str(e), sys.exc_info()))
        sys.exit(1)
