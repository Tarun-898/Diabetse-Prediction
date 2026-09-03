import os, sys


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        self.error_message = CustomException.get_error_details(
            error_message=error_message, error_details=error_details
        )
        super().__init__(self.error_message)

    @staticmethod
    def get_error_details(error_message: Exception, error_details: sys):
        _, _, exc_tb = error_details.exc_info()

        try_block_error_msg = exc_tb.tb_lineno
        exception_block_error_msg = exc_tb.tb_frame.f_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename

        error_message = f"""
error in line block:[{try_block_error_msg}] 
and in exception block:[{exception_block_error_msg}] 
with file name:[{file_name}] error message:[{error_message}]"""
        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"CustomException({self.error_message})"
