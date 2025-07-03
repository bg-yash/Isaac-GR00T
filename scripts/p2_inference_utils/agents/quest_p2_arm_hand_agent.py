from time import sleep, perf_counter

import numpy as np
import rerun as rr
from oculus_reader import OculusReader

from p2_teleop.agents.quest_p2_arm_agent import QuestP2ArmAgent
from p2_teleop.constants import (
    DEFAULT_SERVO_DT,
    MAX_FINGER_DQ,
    DEFAULT_RIGHT_IP,
    CURRENT_VERSION_OBS_HAND,
)
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.p2 import hand_api
from p2_teleop.p2.bg_p2_utils import setup_hand_passthrough, HAND_Q_OPEN
from p2_teleop.p2.p2_math_utils import interp
from p2_teleop.teleop_user_controls import SingleClickCycler, ButtonType
from p2_teleop.observations import ObservationRight
from p2_teleop.actions import ActionRight


class QuestP2ArmHandAgent(QuestP2ArmAgent):
    """
    Class representing P2 arm w/ P2 Hand
    """

    def __init__(
        self,
        which_hand: str,
        which_robot: str,
        event_manager: ButtonEventManager | None,
        oculus_reader: OculusReader | None,
        pos_scale: np.ndarray | None = None,
        ip: str = DEFAULT_RIGHT_IP,
        servo_dt: float = DEFAULT_SERVO_DT,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        # FIXME: update for the new J6 adapter
        tcp_offset = np.array([0.080, 0.0, 0.275, 0, 0, 0])
        payload = np.array([2.72, 0.011, 0.012, 0.145, 0, 0, 0, 0, 0, 0])
        if pos_scale is None:
            pos_scale = np.array([1.5, 1.5, 2.0])
        super().__init__(
            ip=ip,
            which_hand=which_hand,
            which_robot=which_robot,
            pos_scale=pos_scale,
            event_manager=event_manager,
            oculus_reader=oculus_reader,
            servo_dt=servo_dt,
            verbose=verbose,
            tcp_offset=tcp_offset,
            payload=payload,
            dry_run=dry_run,
        )

        if not dry_run:
            setup_hand_passthrough(self.ip)

        self.grasp_qs = self.get_hand_open_closed_qs()
        if self.event_manager and self.oculus_reader:
            self.grasp_cycler = SingleClickCycler(
                self.event_manager,
                ButtonType.PRIMARY,
                self.which_hand,
                self.grasp_qs,
                next_option_cb=self.on_grasp_cycle,
            )
        else:
            self.grasp_cycler = None

    @staticmethod
    def on_grasp_cycle(idx, option):
        print(f"Switching grasp to {idx}")

    @staticmethod
    def get_hand_open_closed_qs():
        FINGER_STOWED = [0, 70, 78]

        return [
            (
                "WIDE_4F_PINCH",
                np.deg2rad(
                    [53] + [0, 2, 2] + [-5, 2, 2] + [-1, 2, 2] + [1, 2, 2] + [5, 2, 2]
                ),
                np.deg2rad(
                    [53]
                    + [0, 33, 3]
                    + [-9, 39, 23]
                    + [-4, 47, 23]
                    + [5, 39, 23]
                    + [9, 39, 23]
                ),
            ),
            (
                "2F_PINCH",
                np.deg2rad(
                    [42]
                    + [-7, 2, 2]
                    + [-8, 8, 7]
                    + [-6, 2, 7]
                    + FINGER_STOWED
                    + FINGER_STOWED
                ),
                np.deg2rad(
                    [42]
                    + [-7, 30, 2]
                    + [-8, 38, 20]
                    + [-6, 42, 20]
                    + FINGER_STOWED
                    + FINGER_STOWED
                ),
            ),
            (
                "1F_Pinch",
                np.deg2rad(
                    [40]
                    + [-10, 2, 2]
                    + [-10, 15, 3]
                    + FINGER_STOWED
                    + FINGER_STOWED
                    + FINGER_STOWED
                ),
                np.deg2rad(
                    [40]
                    + [-7, 28, 2]
                    + [-12, 42, 20]
                    + FINGER_STOWED
                    + FINGER_STOWED
                    + FINGER_STOWED
                ),
            ),
            (
                "BENT_4F_PINCH",
                np.deg2rad(
                    [53]
                    + [0, 2, 2]
                    + [-5, 2, 35]
                    + [-1, 2, 35]
                    + [1, 2, 35]
                    + [5, 2, 35]
                ),
                np.deg2rad(
                    [53]
                    + [0, 25, 3]
                    + [-9, 20, 50]
                    + [-4, 20, 50]
                    + [5, 20, 50]
                    + [9, 20, 50]
                ),
            ),
            (
                "THUMB_PINCH",
                np.deg2rad(
                    [1]
                    + [-14, 2, 2]
                    + [-14, 22, 54]
                    + [-8, 60, 79]
                    + [0, 79, 79]
                    + [0, 79, 79]
                ),
                np.deg2rad(
                    [1]
                    + [-14, 35, 2]
                    + [0, 22, 54]
                    + [-8, 60, 79]
                    + [0, 79, 79]
                    + [0, 79, 79]
                ),
            ),
        ]

    def _actuate_gripper(self, action: ActionRight):
        rr.log("gripper/power_grasp_frac", rr.Scalar(action["power_grasp_frac"]))
        rr.log("gripper/thumb_lateral_frac", rr.Scalar(action["thumb_lateral_frac"]))
        rr.log("gripper/thumb_grasp_frac", rr.Scalar(action["thumb_grasp_frac"]))

        _, current_q = hand_api.getHandJointsPosition(self.ip)

        # TODO: override plan() to compute the grasp_q's so that it can be saved in self.latest_action correctly
        pg_frac = action["power_grasp_frac"]
        _, q_open, q_closed = self.grasp_cycler.get()
        grasp_q = interp(q_open, q_closed, pg_frac)

        # limit the max change in joint angles
        dq = grasp_q - current_q
        dq = np.clip(dq, -MAX_FINGER_DQ, MAX_FINGER_DQ)
        grasp_q_clipped = current_q + dq

        # smooth with current
        grasp_q_smoothed = interp(current_q, grasp_q_clipped, 0.7)
        hand_api.passthroughHandJoints(grasp_q_smoothed, ip=self.ip)

    def clear_errors(self):
        super().clear_errors()
        setup_hand_passthrough(self.ip, init=False)

    def blocking_open_in_passthrough(self):
        timeout = 1
        t0 = perf_counter()
        while True:
            _, current_q = hand_api.getHandJointsPosition(self.ip)
            grasp_q_smoothed = interp(current_q, HAND_Q_OPEN, 0.5)
            hand_api.passthroughHandJoints(grasp_q_smoothed, ip=self.ip)

            q_err = np.max(np.abs(HAND_Q_OPEN - current_q))
            if q_err < np.deg2rad(2):
                break
            if perf_counter() - t0 > timeout:
                return

            sleep(0.01)

    def get_obs(self) -> ObservationRight:
        obs_base = super().get_obs()
        _, current_q = hand_api.getHandJointsPosition(self.ip)
        obs = ObservationRight(**obs_base, hand_q=current_q)
        # TODO: add hand state
        # ret, tactile_data = hand_api.getPerceptualSensorData(self.ip)
        obs["version"] += CURRENT_VERSION_OBS_HAND
        return obs
