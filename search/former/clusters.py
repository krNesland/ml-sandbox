from typing import List, Tuple
import numpy as np


def is_in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < rows and 0 <= y < cols

def get_neighbors(grid: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
    rows, cols = grid.shape
    target_shape = grid[x, y]
    visited = np.zeros_like(grid, dtype=bool)
    cluster = []

    def dfs(x: int, y: int):
        if not is_in_bounds(x, y, rows, cols) or visited[x, y] or grid[x, y] != target_shape:
            return
        visited[x, y] = True
        cluster.append((x, y))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    dfs(x, y)
    return cluster

def get_unique_clusters(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []

    for x in range(rows):
        for y in range(cols):
            if not visited[x, y] and grid[x, y] != 0:
                cluster = get_neighbors(grid, x, y)
                for cx, cy in cluster:
                    visited[cx, cy] = True
                if cluster:
                    clusters.append(cluster)

    return clusters
