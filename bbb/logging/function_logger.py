from functools import wraps

class FunctionLogger:
    """Base class decorator for logging pre- and post-function evaluation.

    Moves a bunch of logging cruft from the main code to the subclassed
    loggers. Example usage:

        import logging
        fnlogger = FunctionLogger(logging)

        @fnlogger
        def func(*args, **kwds):
            ...

    `FunctionLogger.pre()` is called before function evaluation and
    `FunctionLogger.post()` is called after function evaluation. Subclasses
    will have access to the wrapped function's return value through
    `post(return_value)`.
    """

    def __init__(self, logger, prefix=''):
        self.logger = logger
        self.prefix = prefix

    def pre(self, args, kwds):
        pass

    def post(self, args, kwds, func_return):
        pass

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwds):
            self.pre(args, kwds)
            return_value = func(*args, **kwds)
            self.post(args, kwds, return_value)
            return return_value
        return wrapper
