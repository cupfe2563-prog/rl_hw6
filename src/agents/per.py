from src.agents.base_agent import BaseDQNAgent
import torch
from torch import nn
import random

class PERDQNAgent(BaseDQNAgent):
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
        self.alpha = 0.5
        self.beta = 0.5

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

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.online_net(next_state_tensor).argmax().item()
            next_q = self.target_net(next_state_tensor)[next_action]
            y = reward + self.gamma * (1 - float(done)) * next_q
            q = self.online_net(state_tensor)[action]
            priority = (y - q).abs().item() + 1e-3
        self.reply_buffer.append((state, action, reward, next_state, done, priority))
                

    def learn(self):
        if len(self.reply_buffer) < 16:
            return None

        self.online_net.train()
        p = torch.as_tensor([i[-1] ** self.alpha for i in self.reply_buffer],dtype=torch.float32, device=self.device)
        p = p / p.sum()
        ind = torch.multinomial(p, 16, replacement=True).tolist()
        batch = [self.reply_buffer[i] for i in ind]

        weights = p[ind].pow(-self.beta)
        weights /= weights.max()
        losses = []

        for index, weight in zip(ind, weights):
            transition = self.reply_buffer[index]
            state, action, reward, next_state, done, _ = transition

            state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device)
            next_state_tensor = torch.as_tensor(
                next_state, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_action = self.online_net(next_state_tensor).argmax().item()
                next_q = self.target_net(next_state_tensor)[next_action]
                y = reward + self.gamma * (1 - float(done)) * next_q

            q = self.online_net(state_tensor)[action]
            td_error = y - q
            losses.append(weight * td_error.pow(2))

            priority = td_error.detach().abs().item() + 1e-3
            self.reply_buffer[index] = (state, action, reward, next_state, done, priority)

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