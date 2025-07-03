from abc import ABC, abstractmethod
from time import time

import numpy as np

from p2_teleop.agents.quest_arm_agent import QuestArmAgent
from p2_teleop.constants import DEFAULT_SERVO_DT, CURRENT_VERSION_OBS_ARM_AGENT
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.p2 import diana_api, hand_api
from p2_teleop.p2.bg_p2_utils import setup_arm
from p2_teleop.p2.p2_math_utils import pose_to_pos_rot, mat2diana_axangle
from p2_teleop.teleop_user_controls import (
    ButtonType,
    MultiButtonLongPress,
)
from p2_teleop.observations import ObservationArmAgent
from p2_teleop.actions import ActionArmAgent


class QuestP2ArmAgent(QuestArmAgent, ABC):
    """
    Abstract class implementing P2 base functionality.
    Gripper actions are based on the type of arm, e.g. Hand or HFG based grasping.
    """

    def __init__(
        self,
        ip: str,
        which_hand: str,
        which_robot: str,
        pos_scale: np.ndarray,
        event_manager: ButtonEventManager | None,
        oculus_reader,
        servo_dt=DEFAULT_SERVO_DT,
        verbose: bool = False,
        tcp_offset: np.ndarray | None = None,
        payload: np.ndarray | None = None,
        dry_run: bool = False,
    ):
        """

        :param ip:
        :param which_hand:
        :param which_robot:
        :param pos_scale:
        :param event_manager:
        :param oculus_reader:
        :param servo_dt:
        :param verbose:
        :param tcp_offset:
        :param payload: An array [mass, CoM_x, CoM_y, CoM_z, Ixx, Ixy, Ixz, Iyz, Izz] of the payload (hand, hfg, etc)
        :param dry_run:
        """
        super().__init__(
            which_hand=which_hand,
            which_robot=which_robot,
            pos_scale=pos_scale,
            event_manager=event_manager,
            oculus_reader=oculus_reader,
            verbose=verbose,
        )
        self.dry_run = dry_run
        self.ip = ip
        self.tcp_offset = tcp_offset
        self.current_pose = [0] * 6
        self.servo_dt = servo_dt

        if not dry_run:
            setup_arm(self.ip, tcp_offset, overwrite_limits=False, payload=payload)

        if self.event_manager and self.oculus_reader:
            self.reset_event = MultiButtonLongPress(
                {self.which_hand: [ButtonType.PRIMARY, ButtonType.SECONDARY]},
                self.on_reset,
                priority=2,
            )
            self.event_manager.add_event(self.reset_event)

    def get_episode_metadata(self):
        return {
            "ip": self.ip,
            "tcp_offset": self.tcp_offset.tolist(),
            "pos_scale": np.array(self.pos_scale).tolist(),
            "servo_dt": self.servo_dt,
        }

    def get_obs(self) -> ObservationArmAgent:
        assert diana_api.getTcpPos(tcpPose=self.current_pose, ipAddress=self.ip)
        is_collision = diana_api.isCollision()
        robot_state = diana_api.getRobotState().value
        current_q = np.zeros(6)
        diana_api.getJointPos(current_q, ipAddress=self.ip)
        now = time()
        current_pos, current_rot = pose_to_pos_rot(self.current_pose)
        current_qdot = np.zeros(6)
        diana_api.getJointAngularVel(current_qdot, ipAddress=self.ip)
        # NOTE: If you update this dict, also update the version number!
        obs = ObservationArmAgent(
            ee_pos=current_pos,
            ee_rot=current_rot,
            q=current_q,
            qdot=current_qdot,
            t=now,
            is_collision=is_collision,
            robot_state=robot_state,
            version=CURRENT_VERSION_OBS_ARM_AGENT,
        )
        return obs

    def _move_arm(self, action: ActionArmAgent):
        cmd_ee_pos = action["ee_pos"]
        cmd_ee_rot = action["ee_rot"]
        cmd_ee_axis = mat2diana_axangle(cmd_ee_rot)
        pose = np.concatenate((cmd_ee_pos, cmd_ee_axis))

        # TODO: do we need to tune these parameters?
        if not self.dry_run:
            diana_api.servoL(
                pose,
                t=self.servo_dt,
                ah_t=self.servo_dt * 5,
                gain=300,
                scale=1.0,
                ipAddress=self.ip,
            )

    @abstractmethod
    def _actuate_gripper(self, action: ActionArmAgent):
        pass

    def execute(self, action: ActionArmAgent):
        self._move_arm(action)
        self._actuate_gripper(action)

    def step(self):
        obs = self.get_obs()
        action = self.plan(obs)
        if action is not None:
            self.execute(action)
        return obs, action

    def on_reset(self, _):
        """
        Check errors and clear them if the button is pressed

        Button data will be passed in as required by the ButtonEvent callback API, but is not used here
        """
        print("Resetting")
        last_arm_error = diana_api.getLastError(ipAddress=self.ip)
        last_hand_error = hand_api.getHandLastError(ip=self.ip)
        is_error = last_arm_error != 0 or last_hand_error not in [-1, 0]

        if is_error:
            self.clear_errors()

    def clear_errors(self):
        """May be overridden to reset the hand/gripper state as well"""
        assert diana_api.cleanErrorInfo(self.ip)
        assert diana_api.releaseBrake(self.ip)
