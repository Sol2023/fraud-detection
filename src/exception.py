import sys

def error_message_detail(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message ="Error occured in python name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, message:str, error_detail:sys):
        self.message = message
        self.error_detail = error_detail
        error_message_detail(self.message, self.error_detail)
        super().__init__(self.message)

    def __str__(self):
        return self.message