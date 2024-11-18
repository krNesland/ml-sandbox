from typing import List, Tuple

def is_in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < rows and 0 <= y < cols

def get_neighbors(grid: List[List[int]], x: int, y: int) -> List[Tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    target_shape = grid[x][y]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    cluster = []

    def dfs(x: int, y: int):
        if not is_in_bounds(x, y, rows, cols) or visited[x][y] or grid[x][y] != target_shape:
            return
        visited[x][y] = True
        cluster.append((x, y))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    dfs(x, y)
    return cluster

def get_unique_clusters(grid: List[List[int]]) -> List[List[Tuple[int, int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    clusters = []

    for x in range(rows):
        for y in range(cols):
            if not visited[x][y] and grid[x][y] != 0:
                cluster = get_neighbors(grid, x, y)
                for cx, cy in cluster:
                    visited[cx][cy] = True
                if cluster:
                    clusters.append(cluster)

    return clusters
