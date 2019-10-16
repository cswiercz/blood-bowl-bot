import logging
import sys

def get_logger():
    logger = logging.getLogger('bbb')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    # set stream handler for the first time
    formatter = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

