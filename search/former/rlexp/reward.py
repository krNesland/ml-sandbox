"""
Created on 2024-11-30

Author: Kristoffer Nesland

Description: Brief description of the file
"""

import numpy as np


def calc_reward(previous_state: np.ndarray, new_state: np.ndarray) -> float:
    """
    Should in principle be quite similar to giving the agent a large reward when finishing and a small negative reward for each turn. However, it might the simpler to learn from the current implementation.
    """
    n_zeros_new = np.sum(new_state == 0)
    n_zeros_previous = np.sum(previous_state == 0)

    n_more_zeros = n_zeros_new - n_zeros_previous

    return n_more_zeros - 1
