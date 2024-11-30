import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from search.former.rlexp.v0.network import DQN


class DQNAgent:
    def __init__(self, state_size, action_size):
        self._state_size = state_size
        self._action_size = action_size
        self._model = DQN(state_size, action_size)
        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=0.0001,
        )
        self._criterion = nn.MSELoss()
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.1
        self._gamma = 1.0  # Discount factor

    @property
    def model(self) -> nn.Module:
        return self._model

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self._action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self._model(state)
        return np.argmax(q_values.detach().numpy())

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target += self._gamma * torch.max(self._model(next_state)).item()

        state = torch.FloatTensor(state).unsqueeze(0)
        target_f = self._model(state)
        target_f[0][action] = target

        self._optimizer.zero_grad()
        loss = self._criterion(self._model(state), target_f)
        loss.backward()
        self._optimizer.step()
