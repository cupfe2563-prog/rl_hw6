from src.agents.base_agent import BaseDQNAgent
import torch
from torch import nn
import random

class DoubleDQNAgent(BaseDQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim)
        self.reply_buffer = []
        self.epsilon = 1
        self.gamma = 0.9
        self.online_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
        self.learn_steps = 0
        self.target_update_interval = 100
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state, training: bool = True):
            self.online_net.eval()
            self.target_net.eval()
            
            state = torch.as_tensor(state,dtype=torch.float32,device=self.device,)
            with torch.no_grad():
                rewards = self.online_net(state)
           
            act = rewards.argmax(dim=-1).item()
            if training:
                if random.random()<self.epsilon:
                     return random.randrange(self.action_dim)
            return act


    def remember(self, state, action, reward, next_state, done):
        if len(self.reply_buffer) >= 10_000: # тут идея, чтобы сохранять только боле поздний опыт
            self.reply_buffer.pop(0) # ограничить потреблние памяти

        self.reply_buffer.append(
            (state, action, reward, next_state, done))
                

    def learn(self):
        if len(self.reply_buffer) < 16:
            return None

        self.online_net.train()
        batch = random.sample(self.reply_buffer, k=16)
        losses = []

        for state, action, reward, next_state, done in batch:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_action = self.online_net(next_state).argmax().item()
                next_q = self.target_net(next_state)[next_action]
                y = reward + self.gamma * (1 - float(done)) * next_q

            q = self.online_net(state)[action]
            losses.append((y - q).pow(2))

        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def step_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.99995)
