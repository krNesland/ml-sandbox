from typing import List

import numpy as np


def is_in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < rows and 0 <= y < cols


def get_neighbors(grid: np.ndarray, x: int, y: int) -> np.ndarray:
    rows, cols = grid.shape
    target_shape = grid[x, y]
    cluster_mask = np.zeros_like(grid, dtype=bool)

    def dfs(x: int, y: int):
        if (
            not is_in_bounds(x, y, rows, cols)
            or grid[x, y] != target_shape
            or cluster_mask[x, y]
        ):
            return
        cluster_mask[x, y] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    dfs(x, y)
    return cluster_mask


def get_unique_clusters(grid: np.ndarray) -> List[np.ndarray]:
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    cluster_masks = []

    for x in range(rows):
        for y in range(cols):
            if not visited[x, y] and grid[x, y] != 0:
                cluster_mask = get_neighbors(grid, x, y)
                if np.any(cluster_mask):
                    visited[cluster_mask] = True
                    cluster_masks.append(cluster_mask)

    return cluster_masks
