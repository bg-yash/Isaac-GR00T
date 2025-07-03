from abc import ABC, abstractmethod

from p2_teleop.episode_recording.episode import Episode


class Metric(ABC):
    """
    Calculates a values property, which is a numpy array with a leading time dimension.
    Has a function to visualize it in rerun.
    """

    def __init__(self, episode: Episode):
        self.ep = episode

    @abstractmethod
    def viz(self):
        pass
