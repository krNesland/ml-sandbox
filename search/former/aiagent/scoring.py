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

    def __call__(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        unique_shapes = set(np.unique(grid))
        unique_shapes.discard(0)  # Remove the background shape (0)

        total_spread_score = 0

        for shape in unique_shapes:
            shape_indices = np.argwhere(grid == shape)

            if len(shape_indices) < 2:
                # If there are less than two of the shape, the spread is zero
                pass
            else:
                # Calculate pairwise Euclidean distances using broadcasting
                diff = shape_indices[:, np.newaxis, :] - shape_indices[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=-1))

                # Get the upper triangle of the distance matrix, excluding the diagonal
                i_upper = np.triu_indices_from(distances, k=1)
                pairwise_distances = distances[i_upper]

                # Calculate the spread score as the average pairwise distance
                spread_score = np.mean(pairwise_distances)

                total_spread_score += spread_score

            # Add 1 for each shape as it is best to have few unique shapes left
            total_spread_score += 1

        # Negate to make it a minimization problem
        score = -total_spread_score

        return score


class ScoreGridByRL:
    """
    Idea here is to have some ML model trained by RL to give a proxy of the 'position'
    """

    def __call_(self, grid: np.ndarray, cluster_masks: list[np.ndarray]) -> float:
        raise NotImplementedError()
