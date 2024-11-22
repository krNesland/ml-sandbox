import numpy as np


class ScoreBase:
    def __call__(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        raise NotImplementedError()


class ScoreGridByNumRemoved:
    """
    Idea here is that having few remaining non-empty cells is good
    """

    def __call__(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        flattened = grid.flatten()
        return np.sum(flattened == 0) / flattened.size


class ScoreGridByNumClusters:
    """
    Idea here is that having few remaning clusters is good

    Could probably make these functions take clusters as input to speed up computation
    """

    def __call__(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        n_cells = grid.size
        n_clusters = len(cluster_masks)
        return (n_cells - n_clusters) / n_cells


class ScoreGridByShapeAdjecency:
    """
    Idea here is that having identical shapes in adjacency is good
    """

    def __call_(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        raise NotImplementedError()


class ScoreGridByRL:
    """
    Idea here is to have some ML model trained by RL to give a proxy of the 'position'
    """

    def __call_(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        raise NotImplementedError()
