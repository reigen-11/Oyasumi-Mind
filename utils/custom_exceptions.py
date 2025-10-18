import sys

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception=None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
        return f"Error occurred in file: {file_name}, line: {line_number}. \n\n {message} \n error: {error_detail}"
        
    def __eq__(self):
        return self.error_message