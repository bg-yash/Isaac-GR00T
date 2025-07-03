from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseAgent(ABC):
    """Basic interface for the teleop loop."""

    def __init__(
        self,
        event_manager,
    ):
        self.event_manager = event_manager

    def step(self):
        obs = self.get_obs()
        action = self.plan(obs)
        if action is not None:
            self.execute(action)
        return obs, action

    @abstractmethod
    def plan(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray] | None:
        """
        Plans the next action based on the current observation.
        This contains the math for transforming commands from VR to various robot frames.
        It reads the VR buttons and controllers to determine if the control is active.
        It should be run frequently (>50Hz).

        :param obs: The current observation
        :return: The next action to take, or None if no action should be taken.
        """
        pass

    @abstractmethod
    def get_obs(self):
        pass

    @abstractmethod
    def get_episode_metadata(self):
        pass

    @abstractmethod
    def execute(self, action: Dict[str, np.ndarray]):
        """
        Executes the action on the robot.
        This should be run at the robot's control frequency.

        :param action: The action to execute
        """
        pass

    def close(self):
        """
        Call at the end of your program to stop background threads and clean up resources.
        """
        pass
