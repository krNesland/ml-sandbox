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
from search.former.former import Former, convert_to_channeled_grid
from search.former.rlexp.logger import QValuesLogger, TurnsLogger
from search.former.rlexp.reward import calc_reward
from search.former.rlexp.test_boards import ALL_TEST_BOARDS
from search.former.rlexp.v2.agent import DQNAgent

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


if __name__ == "__main__":
    board = ALL_TEST_BOARDS[3]

    org_former = Former.from_board(board)

    agent = DQNAgent(
        n_input_channels=org_former.n_channels,
        n_input_rows=org_former.rows,
        n_input_cols=org_former.cols,
        action_size=org_former.n_cells,
    )
    logger = TurnsLogger()
    board_0_logger = QValuesLogger(board)

    episodes = 5000
    log_every_n_episodes = 10

    with torch.no_grad():
        channeled_grid = convert_to_channeled_grid(
            grid=board_0_logger.board,
            n_channels=org_former.n_channels,
            rows=org_former.rows,
            cols=org_former.cols,
            shapes=org_former.shapes,
        )
        state = torch.FloatTensor(channeled_grid).unsqueeze(0)
        q_values = agent.model(state).detach().numpy()[0]
        board_0_logger.log(q_values=q_values, episode=0)

    for e in range(episodes):
        former = deepcopy(org_former)

        done: bool = False
        turn_num: int = 1
        while not done:
            state = former.channeled_grid
            action = agent.act(state)
            x, y = np.unravel_index(action, (former.rows, former.cols))

            # Remove shapes and apply gravity
            cluster_mask = get_neighbors(former.grid, x, y)
            former.remove_shapes(cluster_mask)
            former.apply_gravity()
            next_state = former.channeled_grid

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
                channeled_grid = convert_to_channeled_grid(
                    grid=board_0_logger.board,
                    n_channels=org_former.n_channels,
                    rows=org_former.rows,
                    cols=org_former.cols,
                    shapes=org_former.shapes,
                )
                state = torch.FloatTensor(channeled_grid).unsqueeze(0)
                q_values = agent.model(state).detach().numpy()[0]
                board_0_logger.log(q_values=q_values, episode=e + 1)

        print(
            f"Episode: {e + 1}/{episodes}, Turns: {turn_num - 1}, Exploration Rate: {agent.epsilon:.2f}"
        )

    logger.plot()
    board_0_logger.plot()
    org_former.print_grid()
