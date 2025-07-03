import numpy as np
import rerun as rr

from p2_teleop.visualization.metric.metric import Metric
from p2_teleop.episode_recording.episode import Episode
from p2_teleop.p2.p2_math_utils import convert_pose_to_torso_frame


class LeftEEPosition(Metric):
    def __init__(self, episode: Episode):
        super().__init__(episode)
        ee_poses = []
        capture_times = []
        for step in self.ep.state_action_data:
            left_state_dict = step["obs"]["left"]
            capture_time = left_state_dict["t"]
            ee_pose = convert_pose_to_torso_frame(
                left_state_dict["ee_pos"], left_state_dict["ee_rot"]
            )
            ee_poses.append(ee_pose)
            capture_times.append(capture_time)
        ee_poses = np.asarray(ee_poses)
        self.capture_times = np.asarray(capture_times)
        self.ee_position = ee_poses[:, :3, 3]

    @property
    def values(self):
        return self.ee_position

    def viz(self):
        capture_time: float
        for i, capture_time in enumerate(self.capture_times):
            rr.set_time_seconds("capture_time", capture_time)
            rr.log(
                "robot/torso/ee_position", rr.LineStrips3D(self.ee_position[: i + 1])
            )
