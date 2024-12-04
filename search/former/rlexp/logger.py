"""
Created on 2024-11-30

Author: Kristoffer Nesland

Description: Brief description of the file
"""

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go


class TurnsLogger:
    def __init__(self):
        self._n_turns: dict[str, list[int]] = {
            "episode": [],
            "n_turns": [],
        }

    def log(self, n_turns: int, episode: int) -> None:
        self._n_turns["episode"].append(episode)
        self._n_turns["n_turns"].append(n_turns)

    def plot(self) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self._n_turns["episode"],
                y=self._n_turns["n_turns"],
                mode="lines+markers",
                name="Number of Turns",
            )
        )
        fig.update_layout(
            title="Number of Turns per Episode",
            xaxis_title="Episode",
            yaxis_title="Number of Turns",
        )
        fig.show()


class QValuesLogger:
    def __init__(self, board: np.ndarray):
        self._q_values: dict[str, list[np.ndarray]] = {
            "episode": [],
            "q_values": [],
        }

        self._board = board

    @property
    def board(self) -> np.ndarray:
        return deepcopy(self._board)

    def log(self, q_values: np.ndarray, episode: int) -> None:
        self._q_values["episode"].append(episode)
        self._q_values["q_values"].append(q_values)

    def plot(self) -> None:
        fig = go.Figure()

        array = np.vstack(self._q_values["q_values"])
        for i in range(array.shape[1]):
            x, y = np.unravel_index(i, self._board.shape)

            fig.add_trace(
                go.Scatter(
                    x=self._q_values["episode"],
                    y=array[:, i],
                    mode="lines+markers",
                    name=f"{i} ({x}, {y}) action Q-value ",
                )
            )
        fig.update_layout(
            title="Q-values per Episode",
            xaxis_title="Episode",
            yaxis_title="Q-value",
        )
        fig.show()
