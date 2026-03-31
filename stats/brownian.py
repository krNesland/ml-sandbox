"""
Brownian motion, like in Einstein's 1905 paper. Even though the mean position of the random walk is (0, 0),
the mean distance from the starting point is proportional to the square root of the number of steps.
"""

import math
import random

import numpy as np
import plotly.graph_objects as go


class Particle:
    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        step_length: float = 1.0,
    ) -> None:
        self._x = x0
        self._y = y0
        self._step_length = step_length

    def step(self):
        angle = random.random() * 2 * math.pi

        dx = self._step_length * math.cos(angle)
        dy = self._step_length * math.sin(angle)

        self._x += dx
        self._y += dy

    def get_pos(self) -> tuple[float, float]:
        return self._x, self._y


if __name__ == "__main__":
    r_fig = go.Figure()
    path_fig = go.Figure()
    plot_every_n = 50

    r_list_list = []

    for i in range(500):
        p = Particle()

        x_list: list[float] = []
        y_list: list[float] = []

        x, y = p.get_pos()
        x_list.append(x)
        y_list.append(y)

        for _ in range(1_000):
            p.step()

            x, y = p.get_pos()
            x_list.append(x)
            y_list.append(y)

        r_list: list[float] = [(x**2 + y**2) ** 0.5 for x, y in zip(x_list, y_list)]
        r_list_list.append(r_list)

        if (i % plot_every_n) == 0:
            r_fig.add_trace(
                go.Scatter(
                    x=list(range(len(r_list))),
                    y=r_list,
                )
            )
            path_fig.add_trace(
                go.Scatter(
                    x=x_list,
                    y=y_list,
                )
            )

    r_np = np.array(r_list_list)
    r_mean = np.mean(r_np, axis=0)

    k = r_mean[-1] / math.sqrt(len(r_mean))
    r_ideal = [k * math.sqrt(x) for x in range(len(r_mean))]

    r_fig.add_trace(
        go.Scatter(
            x=list(range(len(r_mean))),
            y=r_mean,
            line_dash="dash",
        )
    )

    r_fig.add_trace(
        go.Scatter(
            x=list(range(len(r_ideal))),
            y=r_ideal,
            line_dash="dash",
        )
    )

    r_fig.show()
    path_fig.show()
