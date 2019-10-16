from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .agents import Agent

class CartPoleAgent(Agent):

    def __init__(self, model,
                 memory_capacity=5000,
                 batch_size=32,
                 exploration_rate_initial=1.0,
                 exploration_rate_decay=0.999,
                 exploration_rate_final=0.1,
                 exploration_rate_finalized=0.01):
        self.model = model
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size

        self.exploration_rate = exploration_rate_initial
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_final = exploration_rate_final
        self.exploration_rate_finalized = exploration_rate_finalized


    def act(self, environment, with_qvalue=False):
        # determine the action
        action_rewards = self.model.predict(environment.state)
        if np.random.rand() < self.exploration_rate:
            action = environment.action_space.sample()
        else:
            action = action_rewards.argmax()

        # if requested, return the largest qvalue
        if with_qvalue:
            qvalue = action_rewards.max()
            return action, qvalue
        return action


    def remember(self, state, action, reward, next_state, is_terminal):
        state = np.array(state, dtype='float32')
        action = np.array([action], dtype='int64')
        reward = np.array([reward], dtype='float32')
        next_state = np.array(next_state, dtype='float32')
        is_terminal = np.array([is_terminal], dtype='float32')
        self.memory.append((state, action, reward, next_state, is_terminal))


    def update(self):
        batches = torch.utils.data.DataLoader(
            self.memory, batch_size=self.batch_size, drop_last=True,
            shuffle=True)

        for states, action, rewards, next_states, are_terminal in batches:
            self.model.update(states, action, rewards, next_states, are_terminal)
            break  # only process one batch

        if self.exploration_rate > self.exploration_rate_final:
            self.exploration_rate *= self.exploration_rate_decay


    def finalize(self):
        self.explortation_rate = self.exploration_rate_finalized

