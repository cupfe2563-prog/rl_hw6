from src.agents.double_dqn import DoubleDQNAgent
import torch
from torch import nn
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), 0.5 / math.sqrt(in_features)))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), 0.5 / math.sqrt(out_features)))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, state):
        weight = self.weight_mu
        bias = self.bias_mu
        if self.training:
            input_noise = torch.randn(weight.size(1), device=weight.device)
            output_noise = torch.randn(weight.size(0), device=weight.device)
            input_noise = input_noise.sign() * input_noise.abs().sqrt()
            output_noise = output_noise.sign() * output_noise.abs().sqrt()
            weight = weight + self.weight_sigma * output_noise.outer(input_noise)
            bias = bias + self.bias_sigma * output_noise
        return nn.functional.linear(state, weight, bias)


class NoisyNetDQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.epsilon = 0.0
        self.online_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), NoisyLinear(128, action_dim)).to(self.device)
        self.target_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), NoisyLinear(128, action_dim)).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)

    def act(self, state, training: bool = True):
        self.online_net.train(training)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.online_net(state).argmax(dim=-1).item()

    def learn(self):
        self.target_net.train()
        return super().learn()

    def step_epsilon(self):
        pass
