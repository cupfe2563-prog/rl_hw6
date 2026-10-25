from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDQNAgent(ABC):
    device: torch.device
    online_net: nn.Module
    target_net: nn.Module
    epsilon: float
    gamma: float

    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def act(self, state, training: bool = True):
        ...

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        ...

    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def step_epsilon(self):
        ...

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(
            path,
            map_location=self.device,
        )
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)



