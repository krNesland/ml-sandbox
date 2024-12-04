"""
Created on 2024-11-30

Author: Kristoffer Nesland

Description: Train on a single board
"""

import random
from copy import deepcopy

import numpy as np
import torch

from search.former.clusters import get_neighbors
from search.former.former import Former
from search.former.rlexp.logger import QValuesLogger, TurnsLogger
from search.former.rlexp.reward import calc_reward
from search.former.rlexp.test_boards import ALL_TEST_BOARDS
from search.former.rlexp.v1.agent import DQNAgent

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


if __name__ == "__main__":
    board = ALL_TEST_BOARDS[0]

    org_former = Former.from_board(board)

    agent = DQNAgent(
        state_size=org_former.n_cells,
        action_size=org_former.n_cells,
    )
    logger = TurnsLogger()
    board_0_logger = QValuesLogger(board)

    episodes = 500
    log_every_n_episodes = 1

    with torch.no_grad():
        state = torch.FloatTensor(board_0_logger.board.flatten()).unsqueeze(0)
        q_values = agent.model(state).detach().numpy()[0]
        board_0_logger.log(q_values=q_values, episode=0)

    for e in range(episodes):
        former = deepcopy(org_former)

        done: bool = False
        turn_num: int = 1
        while not done:
            state = former.flattened_grid
            action = agent.act(state)
            x, y = np.unravel_index(action, (former.rows, former.cols))

            # Remove shapes and apply gravity
            cluster_mask = get_neighbors(former.grid, x, y)
            former.remove_shapes(cluster_mask)
            former.apply_gravity()
            next_state = former.flattened_grid

            reward = calc_reward(previous_state=state, new_state=next_state)
            done = former.is_grid_empty()

            agent.train(state, action, reward, next_state, done)

            turn_num += 1

        agent.epsilon = max(
            agent.epsilon * (1 - agent.epsilon_decay), agent.epsilon_min
        )

        if e % log_every_n_episodes == 0:
            logger.log(n_turns=turn_num - 1, episode=e + 1)
            with torch.no_grad():
                state = torch.FloatTensor(board_0_logger.board.flatten()).unsqueeze(0)
                q_values = agent.model(state).detach().numpy()[0]
                board_0_logger.log(q_values=q_values, episode=e + 1)

        print(
            f"Episode: {e + 1}/{episodes}, Turns: {turn_num - 1}, Exploration Rate: {agent.epsilon:.2f}"
        )

    logger.plot()
    board_0_logger.plot()
    org_former.print_grid()
