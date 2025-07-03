from typing import TypedDict

import numpy as np


class ObservationWrist(TypedDict):
    name: str
    units: str
    value: float


class ObservationArmAgent(TypedDict):
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    t: float
    is_collision: bool
    robot_state: int
    # Should always be populated with constants.CURRENT_VERSION_OBS_*, concatenating the version of
    # each subsequent subclass in the inheritance chain.
    version: str


class ObservationLeft(ObservationArmAgent, total=False):
    hfg_yaw_position: float
    hfg_yaw_cmd_position: float
    hfg_yaw_cmd_velocity: float
    hfg_yaw_cmd_acceleration: float
    hfg_yaw_current: float
    hfg_yaw_temperature: float
    wrist_pressure_bar: ObservationWrist
    wrist_gripper_proximity: ObservationWrist
    wrist_cup_sensor: ObservationWrist


class ObservationRight(ObservationArmAgent):
    hand_q: np.ndarray


class ObservationWaist(TypedDict):
    waist_yaw: float
    waist_pitch: float
    t: float


class Observation(TypedDict):
    left: ObservationLeft | None
    right: ObservationRight | None
    waist: ObservationWaist | None
