from src.agents.double_dqn import DoubleDQNAgent
from torch import nn
import torch


class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DuelingDQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.online_net = DuelingNet(state_dim, action_dim).to(self.device)
        self.target_net = DuelingNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
