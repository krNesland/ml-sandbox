from typing import Any
from search.former.clusters import get_unique_clusters


class ScoreBase:
    def __call__(self, grid: list[list[int]]) -> float:
        raise NotImplementedError()


class ScoreGridByNumRemoved:
    """
    Idea here is that having few remaining non-empty cells is good
    """
    def __call__(self, grid: list[list[int]]) -> float:
        flattened = [cell for row in grid for cell in row]

        return sum([cell == 0 for cell in flattened]) / len(flattened)
    

class ScoreGridByNumClusters:
    """
    Idea here is that having few remaning clusters is good

    Could praobably make these functions take clusters as input to speed up computation
    """
    def __call__(self, grid: list[list[int]]) -> float:
        n_cells = len([cell for row in grid for cell in row])
        n_clusters = len(get_unique_clusters(grid))
        return (n_cells - n_clusters) / n_cells
