import logging

from .function_logger import FunctionLogger


class EpisodeLogger(FunctionLogger):

    def __init__(self, *args, **kwds):
        self.episode_index = 0
        super(EpisodeLogger, self).__init__(*args, **kwds)


    def post(self, args, kwds, return_value):
        agent, environment = args
        total_reward = return_value
        is_training = kwds.get('with_update', False)
        is_verbose = kwds.get('verbose', False)

        if not is_training:
            self.episode_index = 0

        if is_training and is_verbose:
            self.episode_index += 1
            learning_rate = agent.model.get_learning_rate()
            self.logger.info(f'episode:{self.episode_index}, '
                             f'episode_reward:{total_reward}, ' )

class StepLogger(FunctionLogger):

    def __init__(self, *args, epoch_size=1000, **kwds):
        self.epoch_size = epoch_size
        self.call_count = 0
        self.mean_reward = 0.0
        self.mean_qvalue = 0.0
        super(StepLogger, self).__init__(*args, **kwds)


    def post(self, args, kwds, return_value):
        agent, environment = args
        reward, is_terminal, max_qvalue = return_value

        self.mean_reward += reward
        self.mean_qvalue += max_qvalue
        self.call_count += 1

        if self.call_count % self.epoch_size == 0:
            lr = agent.model.get_learning_rate()
            er = agent.exploration_rate
            self.mean_reward /= self.epoch_size
            self.mean_qvalue /= self.epoch_size
            self.logger.info(f'epoch: {self.call_count}, '
                             f'mean_reward:{self.mean_reward:.6f}, '
                             f'mean_qvalue:{self.mean_qvalue:.6f}, '
                             f'(lr:{lr:.2E}, er:{er:.3f},)')

            self.mean_reward = 0.0
            self.mean_qvalue = 0.0


class ExperimentLogger:

    def __init__(self, logger=None, **logging_basic_config):
        self.logger = logger
        if logger is None:
            logging.basicConfig(**logger_basic_config)
            self.logger = logging

    def log_episode(self, *args, **kwds):
        return EpisodeLogger(self.logger, *args, **kwds)

    def log_step(self, *args, **kwds):
        return StepLogger(self.logger, *args, **kwds)
