import logging
import time

from .function_logger import FunctionLogger

class ValueLogger(FunctionLogger):
    """Decorator for logging the return value of a function."""

    def post(self, args, kwds, return_value):
        self.logger.info(f'{self.prefix}{return_value}')


class CountLogger(FunctionLogger):
    """Decorator for logging the call count of a function."""

    def __init__(self, *args, **kwds):
        self.count = 0
        super(CountLogger, self).__init__(*args, **kwds)

    def pre(self, args, kwds):
        self.count += 1

    def post(self, args, kwds, return_value):
        self.logger.info(f'{self.prefix}{self.count}')


class TimeLogger(FunctionLogger):

    def pre(self, args, kwds):
        self.start = time.time()

    def post(self, args, kwds, return_value):
        elapsed = time.time() - self.start
        self.logger.info(f'{self.prefix}{elapsed}')


class BasicMetricsLogger:
    """Convenient wrapper for basic metrics."""

    def __init__(self, logger=None, **logger_basic_config):
        self.logger = logger
        if logger is None:
            logging.basicConfig(**logger_basic_config)
            self.logger = logging

    def value(self, *args, **kwds):
        return ValueLogger(self.logger, *args, **kwds)

    def count(self, *args, **kwds):
        return CountLogger(self.logger, *args, **kwds)

    def time(self, *args, **kwds):
        return TimeLogger(self.logger, *args, **kwds)

