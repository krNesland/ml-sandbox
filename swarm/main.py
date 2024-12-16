"""
Created 10 April 2021
Kristoffer Nesland, kristoffernesland@gmail.com

Python implementation of https://github.com/SebLague/Slime-Simulation (and some tailoring).
"""

from __future__ import annotations

import multiprocessing as mp
import typing as ty

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage.filters import uniform_filter

from swarm.utils import animation_from_figures

PI: float = 3.1415


class Agent:
    """
    A particle moving around.
    """

    def __init__(self, pos: ty.Tuple[float, float], angle: float):
        self._pos: ty.Tuple[float, float] = pos
        self._angle: float = angle % (2 * PI)

    @property
    def x(self) -> int:
        return int(self._pos[0])

    @property
    def y(self) -> int:
        return int(self._pos[1])

    def sense(
        self,
        sensor_angle_offset: float,
        trail_map: np.ndarray,
        backg_map: ty.Optional[np.ndarray] = None,
        sensor_size: int = 2,
        sensor_distance_offset: float = 5.0,
    ) -> float:
        """
        ...
        """
        sensor_angle: float = self._angle + sensor_angle_offset

        map_ = np.copy(trail_map)
        if backg_map is not None:
            map_ = (map_ + backg_map) / 2.0

        sensor_center: ty.Tuple[int, int] = (
            int(self._pos[0] + np.cos(sensor_angle) * sensor_distance_offset),
            int(self._pos[1] + np.sin(sensor_angle) * sensor_distance_offset),
        )

        left: int = max(0, sensor_center[0] - sensor_size)
        right: int = min(map_.shape[1], sensor_center[0] + sensor_size)
        top: int = max(0, map_.shape[0] - (sensor_center[1] + sensor_size))
        bottom: int = min(
            map_.shape[0], map_.shape[0] - (sensor_center[1] - sensor_size)
        )

        return np.sum(np.sum(map_[top:bottom, left:right]))

    @staticmethod
    def step(
        agent: Agent,
        trail_map: np.ndarray,
        world_width: int,
        world_height: int,
        move_speed: float,
        time_delta: float,
        turn_speed: float,
        backg_map: ty.Optional[np.ndarray] = None,
        sensor_angle_spacing: float = PI / 6,
    ) -> Agent:
        """
        Updating the agent.
        """
        # Update angle.
        weight_forward: float = agent.sense(0.0, trail_map, backg_map)
        weight_left: float = agent.sense(sensor_angle_spacing, trail_map, backg_map)
        weight_right: float = agent.sense(-sensor_angle_spacing, trail_map, backg_map)
        random_steer_strength: float = np.random.rand(1)

        if (weight_forward > weight_left) and (weight_forward > weight_right):
            agent._angle += 0.0
        elif (weight_forward < weight_left) and (weight_forward < weight_right):
            # Turning randomly.
            agent._angle += (2 * random_steer_strength - 1.0) * turn_speed * time_delta
        elif weight_right > weight_left:
            agent._angle -= random_steer_strength * turn_speed * time_delta
        elif weight_left > weight_right:
            agent._angle += random_steer_strength * turn_speed * time_delta

        # Updating the position.
        x_step: float = np.cos(agent._angle) * move_speed * time_delta
        y_step: float = np.sin(agent._angle) * move_speed * time_delta

        new_pos: ty.Tuple[float, float] = (
            agent._pos[0] + x_step,
            agent._pos[1] + y_step,
        )

        # If hitting the wall.
        if (
            (new_pos[0] < 0)
            or (new_pos[0] >= world_width)
            or (new_pos[1] < 0)
            or (new_pos[1] >= world_height)
        ):
            new_pos = (
                min(world_width - 0.01, max(0.0, new_pos[0])),
                min(world_height - 0.01, max(0.0, new_pos[1])),
            )
            agent._angle = (
                np.random.rand(1) * 2 * PI
            )  # Random angle if hitting the wall.

        agent._pos = new_pos

        return agent


class World:
    """
    Can be initialized with a background image. In that case, the agents will tend to follow the high intensity areas
    of this background image, like they tend to follow each other.
    """

    def __init__(
        self,
        width: int,
        height: int,
        move_speed: float = 2.0,
        time_delta: float = 1.0,
        turn_speed: float = 0.5,
        evaporate_speed: float = 2.0,
        diffuse_speed: float = 0.5,
        backg_map: ty.Optional[np.ndarray] = None,
    ):
        if backg_map is not None:
            assert (backg_map.shape[1] == width) and (backg_map.shape[0] == height)

        self.backg_map: ty.Optional[np.ndarray] = backg_map

        self.width: int = width
        self.height: int = height

        self.move_speed: float = move_speed
        self.time_delta: float = time_delta
        self.turn_speed: float = turn_speed
        self.evaporate_speed: float = evaporate_speed
        self.diffuse_speed: float = diffuse_speed
        self.agents: ty.List[Agent] = list()

        # Positive y is downwards for trail_map, but y in world is defined upwards. Therefore, will be some transforms.
        self.trail_map: np.ndarray = np.zeros((height, width), float)

        self.frames: ty.List = list()

        # Multiprocessing.
        self.pool = mp.Pool(mp.cpu_count())

    def __del__(self):
        print("Deleting 'World' object...")
        self.pool.close()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        """
        Iterating the agents.
        """
        if self.n < len(self.agents):
            self.n += 1
            return self.agents[self.n - 1]
        else:
            raise StopIteration

    def draw_frame(self):
        """
        Showing the world with pyplot.
        """
        self.frames.append(np.copy(self.trail_map))

    def add_random_agents(self, num_agents):
        """
        Adding agents with random position and random angle.
        """
        for i in range(num_agents):
            a, b, c = np.random.rand(3)
            self.agents.append(Agent((a * self.width, b * self.height), c * 2 * PI))

    def step(self):
        """
        Advancing all the agents in time and updating the trail map.

        # TODO: A bit slow because of having to copy trail_map and backg_map for every agent. Seems complicated to
        # TODO: fix because they would need to be placed in a shared memory for the processes and this requires quite
        # TODO: a lot of code.
        """
        # Moving the agents.
        self.agents = self.pool.starmap(
            Agent.step,
            [
                (
                    agent,
                    self.trail_map,
                    self.width,
                    self.height,
                    self.move_speed,
                    self.time_delta,
                    self.turn_speed,
                    self.backg_map,
                )
                for agent in self
            ],
        )

        for agent in self:
            self.trail_map[(self.height - 1) - agent.y, agent.x] = 255.0

        # Diffusion.
        blur_map: np.ndarray = uniform_filter(self.trail_map, size=3)
        diffusion: float = self.diffuse_speed * self.time_delta
        diffused_map: np.ndarray = (
            1 - diffusion
        ) * self.trail_map + diffusion * blur_map

        # Combining diffusion and evaporation.
        self.trail_map = np.maximum(
            np.zeros((self.height, self.width), float),
            diffused_map - (self.evaporate_speed * self.time_delta),
        )

    def run(self, num_steps: int = 10):
        """
        Running the simulation.
        """
        for i in range(num_steps):
            print(f"Step {i + 1} / {num_steps}...")
            self.step()
            self.draw_frame()

        figures: list[go.Figure] = list()

        for frame in self.frames:
            rgb_frame = np.stack((frame,) * 3, axis=-1)

            fig = go.Figure()
            fig.add_trace(go.Image(z=rgb_frame))
            figures.append(fig)

        animation = animation_from_figures(figures)
        animation.show()

    @classmethod
    def from_image(cls, im: Image.Image):
        """
        Initializing with an image in the background.
        """
        im_gray = im.convert("L")
        im_np = np.array(im_gray, dtype=float)

        return cls(im_np.shape[1], im_np.shape[0], backg_map=im_np)


if __name__ == "__main__":
    # TODO: Be able to export specific frame.
    # TODO: Copying the numpy arrays all the time probably takes a lot of time.

    world_ = World(400, 300, move_speed=2.0)
    world_.add_random_agents(100)
    world_.run(40)
