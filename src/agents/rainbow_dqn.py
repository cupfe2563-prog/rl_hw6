from src.agents.c51 import C51DQNAgent
from src.agents.noisy_net import NoisyLinear
import torch
from torch import nn


class RainbowNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.features = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, n_atoms))
        self.advantage = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, action_dim * n_atoms))

    def forward(self, state):
        features = self.features(state)
        value = self.value(features).unsqueeze(-2)
        advantage = self.advantage(features).reshape(*state.shape[:-1], self.action_dim, self.n_atoms)
        return value + advantage - advantage.mean(dim=-2, keepdim=True)


class RainbowDQNAgent(C51DQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.online_net = RainbowNet(state_dim, action_dim, self.n_atoms).to(self.device)
        self.target_net = RainbowNet(state_dim, action_dim, self.n_atoms).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
        self.epsilon = 0.0
        self.double = True
        self.alpha = 0.5
        self.beta = 0.5
        self.n_step = 10
        self.temp_buffer = []

    def remember(self, state, action, reward, next_state, done):
        self.temp_buffer.append((state, action, reward, next_state, done))

        while self.temp_buffer and (len(self.temp_buffer) >= self.n_step or done):
            traj = self.temp_buffer[:self.n_step]
            first_state, first_action, *_ = traj[0]
            sum_r = 0.0
            for k, (_, _, current_reward, last_state, last_done) in enumerate(traj):
                sum_r += current_reward * self.gamma ** k
                if last_done:
                    break

            priority = max((transition[-1] for transition in self.reply_buffer), default=1.0)
            if len(self.reply_buffer) >= 10_000:
                self.reply_buffer.pop(0)
            self.reply_buffer.append((first_state, first_action, sum_r, last_state, last_done, k + 1, priority))
            self.temp_buffer.pop(0)

    def learn(self):
        if len(self.reply_buffer) < 16:
            return None

        self.online_net.train()
        self.target_net.train()
        p = torch.as_tensor([transition[-1] ** self.alpha for transition in self.reply_buffer], dtype=torch.float32, device=self.device)
        p = p / p.sum()
        ind = torch.multinomial(p, 16, replacement=True).tolist()
        weights = (len(self.reply_buffer) * p[ind]).pow(-self.beta)
        weights /= weights.max()
        losses = []

        for index, weight in zip(ind, weights):
            transition = self.reply_buffer[index]
            current_loss = self.distribution_loss(*transition[:-1])
            losses.append(weight * current_loss)
            priority = current_loss.detach().item() + 1e-3
            self.reply_buffer[index] = (*transition[:-1], priority)

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.beta = min(1.0, self.beta + 1e-5)
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def step_epsilon(self):
        pass
