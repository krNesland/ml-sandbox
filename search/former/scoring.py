from typing import Any
import numpy as np
from search.former.clusters import get_unique_clusters


class ScoreBase:
    def __call__(self, grid: np.ndarray) -> float:
        raise NotImplementedError()


class ScoreGridByNumRemoved:
    """
    Idea here is that having few remaining non-empty cells is good
    """
    def __call__(self, grid: np.ndarray) -> float:
        flattened = grid.flatten()
        return np.sum(flattened == 0) / flattened.size
    

class ScoreGridByNumClusters:
    """
    Idea here is that having few remaning clusters is good

    Could probably make these functions take clusters as input to speed up computation
    """
    def __call__(self, grid: np.ndarray) -> float:
        n_cells = grid.size
        n_clusters = len(get_unique_clusters(grid))
        return (n_cells - n_clusters) / n_cells
