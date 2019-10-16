import io
import logging
import time

from bbb.logging import BasicMetricsLogger

logger = logging.getLogger('test_logging')
logger.setLevel(logging.INFO)
metrics = BasicMetricsLogger(logger)


class ExampleClass:

    @metrics.value('value:')
    @metrics.time('elapsed:')
    @metrics.count('call-count:')
    def square(self, x):
        time.sleep(0.1)
        return x**2


def test_logger():
    log_capture = io.StringIO()
    log_capture_handler = logging.StreamHandler(log_capture)
    log_capture_handler.setLevel(logging.INFO)
    logger.addHandler(log_capture_handler)

    example = ExampleClass()
    value = example.square(42)

    log_contents = log_capture.getvalue()
    assert 'call-count:1' in log_contents
    assert 'elapsed:0.1' in log_contents
    assert 'value:1764' in log_contents


if __name__ == '__main__':
    test_logger()

