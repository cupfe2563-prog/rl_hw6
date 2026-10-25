from src.agents.double_dqn import DoubleDQNAgent
import torch
from torch import nn
import random


class C51DQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.n_atoms = 51
        self.v_min = -100.0
        self.v_max = 20.0
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.double = False
        self.online_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim * self.n_atoms), nn.Unflatten(-1, (action_dim, self.n_atoms))).to(self.device)
        self.target_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim * self.n_atoms), nn.Unflatten(-1, (action_dim, self.n_atoms))).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)

    def act(self, state, training: bool = True):
        self.online_net.train(training)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probabilities = self.online_net(state).softmax(dim=-1)
            rewards = (probabilities * self.support).sum(dim=-1)
        return rewards.argmax(dim=-1).item()

    def distribution_loss(self, state, action, reward, next_state, done, steps=1):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_probabilities = self.target_net(next_state).softmax(dim=-1)
            action_probabilities = self.online_net(next_state).softmax(dim=-1) if self.double else next_probabilities
            next_action = (action_probabilities * self.support).sum(dim=-1).argmax().item()
            next_distribution = next_probabilities[next_action]
            values = reward + self.gamma ** steps * (1 - float(done)) * self.support
            positions = (values.clamp(self.v_min, self.v_max) - self.v_min) / self.delta
            positions = positions.clamp(0, self.n_atoms - 1)
            lower = positions.floor().long()
            upper = positions.ceil().long()
            projection = torch.zeros_like(self.support)
            projection.index_add_(0, lower, next_distribution * (upper.float() - positions + (lower == upper).float()))
            projection.index_add_(0, upper, next_distribution * (positions - lower.float()))

        log_probabilities = self.online_net(state)[action].log_softmax(dim=-1)
        return -(projection * log_probabilities).sum()

    def learn(self):
        if len(self.reply_buffer) < 16:
            return None

        self.online_net.train()
        self.target_net.eval()
        batch = random.sample(self.reply_buffer, k=16)
        losses = [self.distribution_loss(*transition) for transition in batch]
        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
