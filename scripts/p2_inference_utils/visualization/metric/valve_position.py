import numpy as np
import rerun as rr

from p2_teleop.visualization.metric.metric import Metric
from p2_teleop.episode_recording.episode import Episode


class ValvePosition(Metric):
    def __init__(self, episode: Episode):
        super().__init__(episode)
        valve_positions = []
        capture_times = []
        for step in self.ep.state_action_data:
            if (
                ("actions" not in step)
                or ("left" not in step["actions"])
                or (step["actions"]["left"] is None)
            ):
                break
            valve_positions.append(step["actions"]["left"]["valve_position"])
            capture_times.append(step["obs"]["left"]["t"])
        self.valve_positions = np.asarray(valve_positions)
        self.capture_times = np.asarray(capture_times)

    @property
    def values(self):
        return self.valve_positions

    def viz(self):
        capture_time: float
        for capture_time, valve_position in zip(
            self.capture_times, self.valve_positions
        ):
            rr.set_time_seconds("capture_time", capture_time)
            rr.log("valve_position", rr.Scalar(valve_position))
