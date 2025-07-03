import numpy as np
import rerun as rr

from p2_teleop.visualization.metric.left_ee_velocity import LeftEEVelocity
from p2_teleop.episode_recording.episode import Episode


def smooth(ee_velocity, alpha):
    smoothed = np.zeros_like(ee_velocity)
    smoothed[0] = ee_velocity[0]
    for i in range(1, len(ee_velocity)):
        smoothed[i] = alpha * ee_velocity[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


class LeftEEVelocitySmooth(LeftEEVelocity):
    def __init__(self, episode: Episode):
        super().__init__(episode)
        self.ee_velocity = smooth(self.ee_velocity, alpha=0.1)

    def viz(self):
        capture_time: float
        for capture_time, ee_vel in zip(self.capture_times, self.ee_velocity):
            rr.set_time_seconds("capture_time", capture_time)
            rr.log("l_ee_velocity/smooth", rr.Scalar(ee_vel))
