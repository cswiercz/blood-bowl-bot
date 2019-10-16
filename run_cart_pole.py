#!/usr/bin/env python3

import argparse
import logging
import sys

import bbb
import gym
import numpy as np

from bbb.agents import *
from bbb.experiment import run_experiment
from bbb.models import *
from bbb.logging import get_logger

logger = get_logger()


def main(args):
    logger.info(f'running cart-pole with arguments: {vars(args)}')
    environment = gym.make('CartPole-v1')
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n

    #model = CartPoleDeepQModel(state_dim, action_dim)
    #model = CartPoleDoubleQModel(state_dim, action_dim)
    model = DuelingQModel(state_dim, action_dim)
    agent = CartPoleAgent(model, batch_size=32)
    config = {
        'heatup_episodes': 32,
        'training_episodes': args.training_episodes,
        'evaluation_episodes': args.evaluation_episodes,
        'with_render': args.with_render,
    }

    if args.logfile:
        formatter = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
        file_handler = logging.FileHandler(args.logfile, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    run_experiment(agent, environment, **config)
    environment.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_render', action='store_true', default=False)
    parser.add_argument('--training_episodes', type=int, default=100)
    parser.add_argument('--evaluation_episodes', type=int, default=10)
    parser.add_argument('--logfile', type=str, default='')
    args = parser.parse_args()

    main(args)
