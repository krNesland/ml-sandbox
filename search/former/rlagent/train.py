from copy import deepcopy

import numpy as np

import search.former.boards as boards
from search.former.clusters import get_neighbors
from search.former.former import Former
from search.former.rlagent.agent import DQNAgent

if __name__ == "__main__":
    board = boards.load_board(boards.b_261124)

    rows = 9
    cols = 7
    shapes = [1, 2, 3, 4]  # Use numbers to represent different shapes

    org_former = Former(rows=rows, cols=cols, shapes=shapes)

    agent = DQNAgent(
        state_size=org_former.n_cells,
        action_size=org_former.n_cells,
    )

    episodes = 10_000
    new_board_every = 1

    for e in range(episodes):
        if e % new_board_every == 0:
            # Initializing a new board
            org_former = Former(rows=rows, cols=cols, shapes=shapes)

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

            if former.is_grid_empty():
                reward = 10  # Positive reward for completing the game
                done = True
            else:
                reward = -0.5  # Negative reward for each step

            agent.train(state, action, reward, next_state, done)

            turn_num += 1

        agent.epsilon = max(
            agent.epsilon * (1 - agent.epsilon_decay), agent.epsilon_min
        )

        print(
            f"Episode: {e + 1}/{episodes}, Turns: {turn_num}, Exploration Rate: {agent.epsilon:.2f}"
        )
