import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from search.former.rlagent.network import DQN


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )
        self.criterion = nn.MSELoss()
        self.epsilon = 0.8  # Exploration rate
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target += self.gamma * torch.max(self.model(next_state)).item()

        state = torch.FloatTensor(state).unsqueeze(0)
        target_f = self.model(state)
        target_f[0][action] = target

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(state), target_f)
        loss.backward()
        self.optimizer.step()
