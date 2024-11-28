"""
This is a grid-based puzzle game where the player interacts with a grid filled with random shapes.
The objective is to remove clusters of connected shapes by selecting a shape on the grid.
When a shape is removed, the shapes above it fall down to fill the empty space, similar to gravity.
The player continues to remove shapes until all shapes are removed.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Former:
    def __init__(self, rows: int, cols: int, shapes: list[int]):
        self._rows = rows
        self._cols = cols
        self._shapes = shapes
        self._grid: np.ndarray = self.generate_grid()

    @classmethod
    def from_board(cls, board: np.ndarray) -> "Former":
        rows, cols = board.shape
        shapes = list(np.unique(board))
        instance = cls(rows, cols, shapes)
        instance._grid = board
        return instance

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    # Generate a random grid of shapes
    def generate_grid(self) -> np.ndarray:
        return np.random.choice(self._shapes, size=(self._rows, self._cols))

    # Display the grid
    def print_grid(self):
        for row in self._grid:
            print(" ".join(str(cell) if cell != 0 else "." for cell in row))

    # Plot the grid using plotly with cluster highlighting
    def plot_grid(self, cluster_masks: list[np.ndarray]):
        color_scale = px.colors.qualitative.Plotly

        # Create a text grid based on clusters
        text_grid = [["" for _ in range(self._cols)] for _ in range(self._rows)]
        for cluster_id, cluster_mask in enumerate(cluster_masks):
            coordinates = np.argwhere(cluster_mask)
            for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
                text_grid[x][y] = f"{cluster_id} ({x}, {y})"  # Display cluster ID

        z = [
            [
                None if self._grid[row, col] == 0 else self._grid[row, col]
                for col in range(self._cols)
            ]
            for row in range(self._rows)
        ]

        cols_minus_shapes = self._cols - len(self._shapes)
        assert cols_minus_shapes >= 0

        z.append([None] * self._cols)  # Add an empty row at the bottom
        z.append(
            self._shapes + [None] * cols_minus_shapes
        )  # Trick to always have the colors

        text_grid.append([""] * self._cols)
        text_grid.append(
            [f"{shape}" for shape in self._shapes] + [""] * cols_minus_shapes
        )

        # Plot using plotly
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                colorscale=color_scale,
                showscale=False,
                text=text_grid,
                hoverinfo="text",
                texttemplate="%{text}",  # Show text on the grid
            )
        )
        fig.update_layout(
            title="Grid of Shapes with Clusters",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                autorange="reversed",
            ),
        )
        fig.show()

    # Remove connected shapes
    def remove_shapes(self, cluster_mask: np.ndarray):
        self._grid[cluster_mask] = 0

    # Shift shapes down after removal
    def apply_gravity(self):
        for col in range(self._cols):
            stack = self._grid[:, col][self._grid[:, col] != 0]
            self._grid[:, col] = 0
            self._grid[self._rows - len(stack) :, col] = stack

    # Check if the grid is empty
    def is_grid_empty(self) -> bool:
        return np.all(self._grid == 0)
