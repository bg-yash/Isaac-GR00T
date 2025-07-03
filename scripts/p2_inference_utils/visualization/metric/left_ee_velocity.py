import numpy as np
import rerun as rr

from p2_teleop.visualization.metric.left_ee_position import LeftEEPosition
from p2_teleop.episode_recording.episode import Episode


class LeftEEVelocity(LeftEEPosition):
    def __init__(self, episode: Episode):
        super().__init__(episode)
        times = [step["time"] for step in self.ep.state_action_data]
        dts = np.diff(times)
        ee_velocity = np.diff(self.ee_position, axis=0) / dts[:, None]
        self.ee_velocity = np.linalg.norm(ee_velocity, axis=-1)

    @property
    def values(self):
        return self.ee_velocity

    def viz(self):
        capture_time: float
        for capture_time, ee_vel in zip(self.capture_times, self.ee_velocity):
            rr.set_time_seconds("capture_time", capture_time)
            rr.log("l_ee_velocity/raw", rr.Scalar(ee_vel))
