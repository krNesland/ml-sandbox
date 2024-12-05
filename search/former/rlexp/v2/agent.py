"""
Created on 2024-12-01

Author: Kristoffer Nesland

Description: Brief description of the file
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from search.former.rlexp.v2.network import DQN


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, experience: tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.memory.append(experience)

    def sample(
        self, batch_size: int
    ) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        n_input_channels: int,
        n_input_rows: int,
        n_input_cols: int,
        action_size: int,
        memory_capacity: int = 10_000,
        batch_size: int = 1024,
        target_update_frequency: int = 100,
    ):
        self._n_input_channels = n_input_channels
        self._n_input_rows = n_input_rows
        self._n_input_cols = n_input_cols
        self._action_size = action_size
        self._model = DQN(
            n_input_channels=n_input_channels,
            n_input_rows=n_input_rows,
            n_input_cols=n_input_cols,
            output_dim=action_size,
        )
        self._target_model = DQN(
            n_input_channels=n_input_channels,
            n_input_rows=n_input_rows,
            n_input_cols=n_input_cols,
            output_dim=action_size,
        )
        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=0.00001,
        )
        self._criterion = nn.MSELoss()
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.1
        self._gamma = 1.0  # Discount factor
        self._memory = ReplayMemory(memory_capacity)
        self._batch_size = batch_size
        self._target_update_frequency = target_update_frequency
        self._steps = 0  # To keep track of the number of steps

        # Initialize target model weights to equal the model weights
        self.update_target_model()

    @property
    def model(self) -> nn.Module:
        return self._model

    def update_target_model(self) -> None:
        self._target_model.load_state_dict(self._model.state_dict())

    def act(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self._action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self._model(state)
        return np.argmax(q_values.detach().numpy())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._memory.push((state, action, reward, next_state, done))

    def replay(self) -> None:
        if len(self._memory) < self._batch_size:
            return

        # Sample a minibatch from memory
        minibatch = self._memory.sample(self._batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to a single numpy array first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Then convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute target values
        with torch.no_grad():
            next_q_values = self._target_model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            targets = rewards + (1 - dones) * self._gamma * max_next_q_values

        # Compute current Q values
        q_values = self._model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self._criterion(q_values, targets)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Update target model weights every fixed number of steps
        self._steps += 1
        if self._steps % self._target_update_frequency == 0:
            print(f"Step {self._steps}. Updating target model weights...")
            self.update_target_model()

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay()
