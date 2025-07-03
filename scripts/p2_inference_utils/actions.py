from typing import TypedDict

import numpy as np


class ControllerInputs(TypedDict):
    power_grasp_frac: float
    thumb_lateral_frac: float
    thumb_grasp_frac: float
    A: bool
    B: bool
    X: bool
    Y: bool


class ActionArmAgent(ControllerInputs):
    """Actions for an arm agent

    NOTE: It's a bit strange that this inherets from ControllerInputs, but I wanted to preserve the
    behavior of the existing code more than I wanted the type hinting to make sense.
    """

    ee_pos: np.ndarray
    ee_rot: np.ndarray
    t: float


class ActionLeft(ActionArmAgent, total=False):
    valve_position: float
    hfg_yaw_stop_rotate: bool
    hfg_yaw_cmd_vel: float
    hfg_yaw_goal_yaw: float


class ActionRight(ActionArmAgent):
    pass


class ActionWaist(TypedDict):
    pass


class Action(TypedDict):
    left: ActionLeft | None
    right: ActionRight | None
    waist: ActionWaist | None
