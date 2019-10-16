import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR


class Model:
    """The agent's underlying model and rules on how to update."""

    def build_net(state_dim, action_dim, *args, **kwds):
        """Returns the neural network model used by the RL model."""
        raise NotImplementedError


    def predict(self, state):
        """Returns action rewards for the given state."""
        raise NotImplementedError


    def update(self, states, actions, rewards, next_states, are_terminal):
        """Update the model using a batch of data.

        Parameters
        ----------
        state
            An environment state.
        action
            The action taken by this agent based on the state.
        reward
            Reward received for taking this action.
        next_state
            The resulting state after taking the action.
        is_terminal
            If `True`, indicates that resulting next state is the final state
            of the episode.

        Returns
        -------
        action_qvalues
            The model values at the current states with the chosen actions.
        """
        raise NotImplementedError



class DeepQModel(Model):
    """A basic Deep Q model, as described in [1].

    [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement
        learning." arXiv preprint arXiv:1312.5602 (2013).
    """

    def __init__(self, state_dim, action_dim, discount_factor=0.99):
        self.discount_factor = discount_factor
        self.target_net = self.build_net(state_dim, action_dim)
        self.net = self.build_net(state_dim, action_dim)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=2.5e-4, momentum=0.95)
        self.loss_function = nn.MSELoss()


    def predict(self, state):
        self.net.eval()
        state = torch.tensor([state], dtype=torch.float32, requires_grad=False)
        qvalues = self.net(state).detach().numpy().flatten()
        return qvalues


    def get_learning_rate(self):
        for params in self.optimizer.param_groups:
            return params['lr']


    def update(self, states, actions, rewards, next_states, are_terminal):
        self.net.train()
        batch_size = len(states)

        # compute the q-values from the current states corresponding to the
        # given actions
        current_qvalues = self.net(states).gather(1, actions)

        # compute target q-values from next states corresponding to the
        # maximally beneficial actions
        next_qvalues = self.net(next_states).max(1, keepdim=True)[0]
        expected_qvalues = rewards + (1-are_terminal)*self.discount_factor*next_qvalues

        loss = self.loss_function(current_qvalues, expected_qvalues.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_qvalues


class DoubleQModel(DeepQModel):
    """The Double Q model as described in [2].

    The primary difference is the maintenence of two copies of the network
    parameters, the online network and the target network, where the online
    network is used action selection via the greedy policy and the target
    network is used for evaluation.

    This model appears to have replaced [1] as the "basic deep Q-learning
    model" in the literature.

    [2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement
        learning with double q-learning." Thirtieth AAAI conference on
        artificial intelligence. 2016.

    """

    def __init__(self, *args, **kwds):
        super(DoubleQModel, self).__init__(*args, **kwds)

        self.target_net = self.build_net(*args)
        self._update_target_model(1)  # copy parameters


    def _update_target_model(self, tau):
        target_parameters = self.target_net.parameters()
        parameters = self.net.parameters()
        for target_theta, theta in zip(target_parameters, parameters):
            target_theta.data.copy_(tau*theta + (1 - tau)*target_theta)


    def update(self, states, actions, rewards, next_states, are_terminal):
        self.net.train()
        self.target_net.eval()
        batch_size = len(states)

        # compute current q-values from the current states corresponding to the
        # given actions using the online network
        current_qvalues = self.net(states).gather(1, actions)

        # compute target q-values from next states corresponding to the
        # maximally beneficial actions. Use the online network to determine
        # which action should be taken and the target network to evaluate.
        maximal_actions = self.net(next_states).argmax(1, keepdim=True).detach()
        maximal_qvalues = self.target_net(next_states).gather(1, maximal_actions)
        expected_qvalues = rewards + (1-are_terminal)*self.discount_factor*maximal_qvalues

        loss = self.loss_function(current_qvalues, expected_qvalues.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_model(0.05)
        return current_qvalues



class DuelingQNet(nn.Module):

    def __init__(self, state_dim, action_dim, **kwds):
        super(DuelingQNet, self).__init__(**kwds)

        self.base_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

        self.advantage_net = nn.Sequential(
            nn.Linear(32, action_dim))

        self.value_net = nn.Sequential(
            nn.Linear(32, 1))


    def forward(self, x):
        y = self.base_net(x)
        y_advantage = self.advantage_net(y)
        y_value = self.value_net(y)
        q_value = y_value + y_advantage - y_advantage.mean(1, keepdim=True)
        return q_value


class DuelingQModel(DoubleQModel):

    def __init__(self, *args, **kwds):
        super(DuelingQModel, self).__init__(*args, **kwds)

    def build_net(self, state_dim, action_dim, *args, **kwds):
        return DuelingQNet(state_dim, action_dim)
