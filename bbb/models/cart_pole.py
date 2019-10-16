import torch.nn as nn

from .models import DeepQModel, DoubleQModel


class CartPoleDeepQModel(DeepQModel):

    def build_net(self, state_dim, action_dim):
        net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        return net


class CartPoleDoubleQModel(DoubleQModel):

    def build_net(self, state_dim, action_dim):
        net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        return net

