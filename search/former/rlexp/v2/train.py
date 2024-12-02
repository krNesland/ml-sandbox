import random
from copy import deepcopy

import numpy as np
import torch

from search.former.clusters import get_neighbors
from search.former.former import Former
from search.former.rlexp.logger import QValuesLogger, TurnsLogger
from search.former.rlexp.reward import calc_reward
from search.former.rlexp.test_boards import ALL_TEST_BOARDS
from search.former.rlexp.v2.agent import DQNAgent

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


if __name__ == "__main__":
    rows = 9
    cols = 7
    shapes = [1, 2, 3, 4]  # Use numbers to represent different shapes

    org_former = Former(rows=rows, cols=cols, shapes=shapes)

    agent = DQNAgent(
        state_size=org_former.n_cells,
        action_size=org_former.n_cells,
    )
    logger = TurnsLogger()
    board_0_logger = QValuesLogger(board=ALL_TEST_BOARDS[0])

    episodes = 1_000
    new_board_every = 5
    log_every_n_episodes = 1

    with torch.no_grad():
        state = torch.FloatTensor(board_0_logger.board).unsqueeze(0)
        q_values = agent.model(state).detach().numpy()[0]
        board_0_logger.log(q_values=q_values, episode=0)

    for e in range(episodes):
        if e % new_board_every == 0:
            # Initializing a new board
            org_former = Former(rows=rows, cols=cols, shapes=shapes)

        former = deepcopy(org_former)

        done: bool = False
        turn_num: int = 1
        while not done:
            state = former.grid
            action = agent.act(state)
            x, y = np.unravel_index(action, (former.rows, former.cols))

            # Remove shapes and apply gravity
            cluster_mask = get_neighbors(former.grid, x, y)
            former.remove_shapes(cluster_mask)
            former.apply_gravity()
            next_state = former.grid

            reward = calc_reward(former, x, y)
            done = former.is_grid_empty()

            agent.train(state, action, reward, next_state, done)

            turn_num += 1

        agent.epsilon = max(
            agent.epsilon * (1 - agent.epsilon_decay), agent.epsilon_min
        )

        if e % log_every_n_episodes == 0:
            logger.log(n_turns=turn_num, episode=e + 1)
            with torch.no_grad():
                state = torch.FloatTensor(board_0_logger.board).unsqueeze(0)
                q_values = agent.model(state).detach().numpy()[0]
                board_0_logger.log(q_values=q_values, episode=e + 1)

        print(
            f"Episode: {e + 1}/{episodes}, Turns: {turn_num}, Exploration Rate: {agent.epsilon:.2f}"
        )

    logger.plot()
    board_0_logger.plot()
