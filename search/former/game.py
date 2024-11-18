"""
This is a grid-based puzzle game where the player interacts with a grid filled with random shapes.
The objective is to remove clusters of connected shapes by selecting a shape on the grid.
When a shape is removed, the shapes above it fall down to fill the empty space, similar to gravity.
The player continues to remove shapes until all shapes are removed.

Will work on speeding up later.
"""

import random
import plotly.graph_objects as go
import plotly.express as px


class Former:
    def __init__(self, rows: int, cols: int, shapes: list[int]):
        self._rows = rows
        self._cols = cols
        self._shapes = shapes
        self._grid: list[list[int]] = self.generate_grid()

    @classmethod
    def from_board(cls, board: list[list[int]]) -> 'Former':
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        shapes = list({cell for row in board for cell in row})
        instance = cls(rows, cols, shapes)
        instance._grid = board
        return instance

    @property
    def grid(self):
        return self._grid

    # Generate a random grid of shapes
    def generate_grid(self)-> list[list[int]]:
        return [[random.choice(self._shapes) for _ in range(self._cols)] for _ in range(self._rows)]

    # Display the grid
    def print_grid(self, clusters: list[list[tuple[int, int]]]):
        for row in self._grid:
            print(" ".join(str(cell) if cell != 0 else "." for cell in row))

    # Plot the grid using plotly with cluster highlighting
    def plot_grid(self, clusters: list[list[tuple[int, int]]]):
        color_scale = px.colors.qualitative.Plotly

        # Create a text grid based on clusters
        text_grid = [['' for _ in range(self._cols)] for _ in range(self._rows)]
        for cluster_id, cluster in enumerate(clusters):
            for x, y in cluster:
                text_grid[x][y] = f"{cluster_id} ({x}, {y})"  # Display cluster ID

        z = [[None if self._grid[row][col] == 0 else self._grid[row][col] for col in range(self._cols)] for row in range(self._rows)]

        cols_minus_shapes = self._cols - len(self._shapes)
        assert cols_minus_shapes >= 0
        
        z.append([None] * self._cols)  # Add an empty row at the bottom
        z.append(self._shapes + [None] * cols_minus_shapes)  # Trick to always have the colors

        text_grid.append([''] * self._cols)
        text_grid.append([f"{shape}" for shape in self._shapes] + [''] * cols_minus_shapes)


        # Plot using plotly
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale=color_scale,
            showscale=False,
            text=text_grid,
            hoverinfo='text',
            texttemplate="%{text}"  # Show text on the grid
        ))
        fig.update_layout(
            title='Grid of Shapes with Clusters',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed')
        )
        fig.show()

    # Check if coordinates are within the grid
    def is_in_bounds(self, x, y):
        return 0 <= x < self._rows and 0 <= y < self._cols

    # Get all neighboring shapes of the same type
    def get_neighbors(self, x, y) -> list[tuple[int, int]]:
        target_shape = self._grid[x][y]
        visited = [[False for _ in range(self._cols)] for _ in range(self._rows)]
        cluster = []

        def dfs(x, y):
            if not self.is_in_bounds(x, y) or visited[x][y] or self._grid[x][y] != target_shape:
                return
            visited[x][y] = True
            cluster.append((x, y))
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            for dx, dy in directions:
                dfs(x + dx, y + dy)

        dfs(x, y)
        return cluster

    # Get all unique clusters of the same shape
    def get_unique_clusters(self):
        visited = [[False for _ in range(self._cols)] for _ in range(self._rows)]
        clusters = []

        for x in range(self._rows):
            for y in range(self._cols):
                if not visited[x][y] and self._grid[x][y] != 0:
                    cluster = self.get_neighbors(x, y)
                    for cx, cy in cluster:
                        visited[cx][cy] = True
                    if cluster:
                        clusters.append(cluster)

        return clusters

    # Remove connected shapes
    def remove_shapes(self, x, y):
        cluster = self.get_neighbors(x, y)
        for cx, cy in cluster:
            self._grid[cx][cy] = 0

    # Shift shapes down after removal
    def apply_gravity(self):
        for col in range(self._cols):
            stack = [self._grid[row][col] for row in range(self._rows) if self._grid[row][col] != 0]
            for row in range(self._rows - len(stack)):
                self._grid[row][col] = 0
            for row in range(len(stack)):
                self._grid[self._rows - len(stack) + row][col] = stack[row]

    # Check if the grid is empty
    def is_grid_empty(self):
        return all(cell == 0 for row in self._grid for cell in row)
