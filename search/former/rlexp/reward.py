"""
Created on 2024-11-30

Author: Kristoffer Nesland

Description: Brief description of the file
"""

from search.former.former import Former


def calc_reward(former: Former, x: int, y: int) -> float:
    reward = 0.0

    if former.is_grid_empty():
        reward = 100  # Positive reward for completing the game
    elif former.grid[x, y] == 0:
        reward = -5  # Negative reward for selecting an empty cell
    else:
        reward = -1  # Negative reward for each step

    return reward
