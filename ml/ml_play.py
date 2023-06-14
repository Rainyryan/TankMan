import os
import random
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Box
from stable_baselines3 import PPO

is_debug = False


WIDTH = 1000
HEIGHT = 600
COMMAND = [
    ["NONE"],
    ["FORWARD"],
    ["BACKWARD"],
    ["TURN_LEFT"],
    ["TURN_RIGHT"],
]


def normalize_obs(obs: np.ndarray, observation_space: Box) -> np.ndarray:
    return (obs - observation_space.low) / (
        observation_space.high - observation_space.low
    )


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        print(f"Initial Game {ai_name} ml script")
        self.player = ai_name

        self.observation_space = Box(
            low=0, high=np.array([len(COMMAND), WIDTH, HEIGHT, 360, WIDTH, HEIGHT])
        )
        self.model = PPO.load(os.path.join(os.path.dirname(__file__), "resupply_model"))

        self.prev_action = None
        self.prev_obs = None
        self.supply_type = "oil_stations"

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        obs = normalize_obs(self.get_obs(scene_info), self.observation_space)
        prev_obs = self.prev_obs if self.prev_obs is not None else obs
        action, _ = self.model.predict(np.concatenate([obs, prev_obs]))

        self.prev_action = action
        self.prev_obs = obs

        return COMMAND[action]

    def reset(self):
        """
        Reset the status
        """
        self.prev_action = None
        self.prev_obs = None

    def get_obs(
        self,
        scene_info: dict,
    ) -> np.ndarray:
        # Function to calculate the quadrant of the given coordinate
        def calculate_quadrant(x: int, y: int) -> int:
            mid_x = WIDTH // 2
            mid_y = (HEIGHT - 100) // 2
            return (
                1
                if x >= mid_x and y < mid_y
                else 2
                if x < mid_x and y < mid_y
                else 3
                if x < mid_x and y >= mid_y
                else 4
            )

        # Function to clip values between lower and upper bounds
        clip = lambda x, l, u: max(min(x, u), l)

        # Previous action
        prev_action = self.prev_action or 0

        # Player info
        x = clip(scene_info["x"], 0, WIDTH)
        y = clip(scene_info["y"], 0, HEIGHT)
        angle = (scene_info["angle"] + 360) % 360
        player_quadrant = calculate_quadrant(x, y)

        # Supply station info
        supply_stations = scene_info[self.supply_type + "_info"]
        supply_x, supply_y = None, None
        for station in supply_stations:
            supply_x, supply_y = station["x"], station["y"]

            # Make sure the supply station is in the same side as the player's location
            supply_quadrant = calculate_quadrant(supply_x, supply_y)
            if (supply_quadrant in (1, 4) and player_quadrant in (1, 4)) or (
                supply_quadrant in (2, 3) and player_quadrant in (2, 3)
            ):
                break

        assert supply_x is not None and supply_y is not None

        obs = np.array([prev_action, x, y, angle, supply_x, supply_y], dtype=np.float32)
        return obs
