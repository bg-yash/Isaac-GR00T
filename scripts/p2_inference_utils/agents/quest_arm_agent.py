from abc import ABC
from enum import auto, Enum
from time import time
from typing import Dict

import numpy as np
# import rerun as rr
# from oculus_reader import OculusReader
from transforms3d import euler

from p2_teleop.agents.base_agent import BaseAgent
# from p2_teleop.p2 import waist_api
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.static_transforms import (
    torso2left_pos,
    torso2right_pos,
    torso2left,
    torso2right,
    vr2base,
    vr2cam,
    cam2ee_left,
    cam2ee_right,
)
from p2_teleop.teleop_user_controls import ButtonType, SingleClickCycler
from p2_teleop.actions import ActionArmAgent, ControllerInputs
from p2_teleop.observations import ObservationArmAgent


class ControlFrame(Enum):
    BASE = auto()
    EE = auto()


class ControlState(Enum):
    INACTIVE = auto()
    ACTIVE = auto()


def get_side_name(which_hand):
    match which_hand:
        case "l":
            return "left"
        case "r":
            return "right"


class QuestArmAgent(BaseAgent, ABC):

    def __init__(
        self,
        which_hand: str,
        which_robot: str,
        pos_scale,
        event_manager: ButtonEventManager | None,
        oculus_reader: None,
        verbose: bool = False,
    ) -> None:
        """

        :param which_hand: "l" or "r"
        :param which_robot: "l" or "r". Should be the opposite of which_hand if you're facing the robot.
        :param event_manager:
        :param oculus_reader:
        :param verbose:
        :param pos_scale: scales the delta position commands
        """
        super().__init__(event_manager)

        self.which_hand = which_hand
        self.which_robot = which_robot
        self.event_manager = event_manager
        self.oculus_reader = oculus_reader

        if isinstance(pos_scale, (int, float)):
            self.pos_scale = np.array([pos_scale, pos_scale, pos_scale])
        elif isinstance(pos_scale, list) or isinstance(pos_scale, tuple):
            self.pos_scale = np.array([float(s) for s in pos_scale])
        else:
            self.pos_scale = pos_scale
        print(f"{str(self.__class__)} using pos_scale: {pos_scale}")

        assert self.which_hand in ["l", "r"]
        assert self.which_robot in ["l", "r"]

        if self.event_manager:
            self.control_frame_switcher = SingleClickCycler(
                event_manager,
                ButtonType.SECONDARY,
                which_hand,
                options=list(ControlFrame),
                next_option_cb=self.on_cf_cycle,
            )
        else:
            print("No event manager provided. Quest controller Buttons will not work.")

        self.oculus_reader = oculus_reader
        if oculus_reader is None:
            print("No Oculus reader provided. Quest controller will not work.")

        self._verbose = verbose
        self.error_msg_count = 0

        self.trigger_state = {"l": False, "r": False}
        self.control_state = ControlState.INACTIVE
        self.activation_countdown_max = 100
        self.activation_countdown = self.activation_countdown_max
        self.reference_vr_pose = None
        self.reference_ee_rot_robot = None
        self.reference_ee_pos_robot = None

    def on_cf_cycle(self, idx, option):
        print(f"Switching control frame to {option}")

        obs = self.get_obs()
        _, pose_key, _, _ = self.get_oculus_data_keys_for_side()
        pose_data, button_data = self.oculus_reader.get_transformations_and_buttons()
        is_valid, error_msg = self.has_valid_controller_data(pose_data, button_data)
        if not is_valid:
            if self._verbose:
                print(error_msg)
            return None

        base2side_pos, base2side, ee_pos_robot, ee_rot_robot = (
            self.get_transforms_for_side(obs)
        )

        self.reference_vr_pose = pose_data[pose_key]
        self.reference_ee_rot_robot = ee_rot_robot
        self.reference_ee_pos_robot = ee_pos_robot

    def on_control_deactivated(self):
        pass

    def plan(self, obs: ObservationArmAgent) -> ActionArmAgent | None:
        """
        Plans the next action based on the current observation.
        This contains the math for transforming commands from VR to various robot frames.
        It reads the VR buttons and controllers to determine if the control is active.
        It should be run frequently (>50Hz).

        :param obs: The current observation
        :return: The next action to take, or None if no action should be taken.
        """
        joystick_key, pose_key, power_grasp_key, trigger_key = (
            self.get_oculus_data_keys_for_side()
        )

        # check the trigger button state
        pose_data, button_data = self.oculus_reader.get_transformations_and_buttons()
        is_valid, error_msg = self.has_valid_controller_data(pose_data, button_data)

        if not is_valid:
            if self.error_msg_count > 2_000:
                self.error_msg_count = 0
                print(error_msg)
            self.error_msg_count += 1
            self.control_state = ControlState.INACTIVE
            self.activation_countdown = self.activation_countdown_max
            return None

        current_vr_pose = pose_data[pose_key]
        controller_pos_in_vr = current_vr_pose[:3, 3]
        controller_rot_in_vr = current_vr_pose[:3, :3]
        rr.log(
            f"robot/vr/{self.which_hand}_controller",
            rr.Transform3D(
                translation=controller_pos_in_vr,
                mat3x3=controller_rot_in_vr,
                axis_length=0.1,
            ),
        )

        controller_inputs = self.get_controller_inputs(
            button_data, power_grasp_key, joystick_key
        )

        base2side_pos, base2side, ee_pos_robot, ee_rot_robot = (
            self.get_transforms_for_side(obs)
        )

        self.trigger_state[self.which_hand] = button_data[trigger_key][0] > 0.5
        if self.trigger_state[self.which_hand]:
            if self.control_state == ControlState.INACTIVE:
                if self.activation_countdown == self.activation_countdown_max:
                    print(f"setting VR reference to {self.reference_vr_pose}")
                self.reference_vr_pose = pose_data[pose_key]
                self.reference_ee_rot_robot = ee_rot_robot
                self.reference_ee_pos_robot = ee_pos_robot

                self.activation_countdown -= 1
                if self.activation_countdown <= 0:
                    self.activation_countdown = self.activation_countdown_max
                    self.control_state = ControlState.ACTIVE
                return None
            else:
                rr.log(
                    f"robot/{self.which_robot}_ee",
                    rr.Transform3D(
                        translation=ee_pos_robot, mat3x3=ee_rot_robot, axis_length=0.1
                    ),
                )

                rr.log(
                    f"robot/vr",
                    rr.Transform3D(
                        translation=[0, 0.7, 0], mat3x3=vr2base, axis_length=0.1
                    ),
                )
                rr.log(
                    f"robot/vr/{self.which_hand}_reference",
                    rr.Transform3D(
                        translation=self.reference_vr_pose[:3, 3],
                        mat3x3=self.reference_vr_pose[:3, :3],
                        axis_length=0.1,
                    ),
                )

                # Compute the delta pos/rot in VR frame
                delta_rot = controller_rot_in_vr @ self.reference_vr_pose[:3, :3].T
                delta_pos = controller_pos_in_vr - self.reference_vr_pose[:3, 3]

                match self.control_frame_switcher.get():
                    case ControlFrame.EE:
                        if self.which_robot == "r":
                            cam2ee = cam2ee_right
                        else:
                            cam2ee = cam2ee_left
                        vr2ee = cam2ee @ vr2cam

                        rr.log(
                            f"robot/vr/delta_pos",
                            rr.LineStrips3D([[0, 0, 0], delta_pos]),
                        )

                        # Rotate the delta from VR frame to robot EE frame
                        delta_rot_ee = vr2ee @ delta_rot @ vr2ee.T
                        delta_pos_ee = vr2ee @ delta_pos

                        # Transform the delta from EE frame to robot base frame
                        delta_rot_robot = ee_rot_robot @ delta_rot_ee @ ee_rot_robot.T
                        delta_pos_robot = ee_rot_robot @ delta_pos_ee

                    case ControlFrame.BASE:
                        # Transform the delta from VR frame to Robot base frame
                        delta_rot_robot = vr2base @ delta_rot @ vr2base.T
                        delta_pos_robot = vr2base @ delta_pos
                    case _:
                        raise NotImplementedError(
                            f"Unknown control frame: {self.control_frame_switcher.get()}"
                        )

                # Apply the delta to the reference EE pose. These are in base frame.
                delta_pos_robot_scaled = self.pos_scale * delta_pos_robot
                next_ee_pos_robot = delta_pos_robot_scaled + self.reference_ee_pos_robot
                next_ee_rot_robot = delta_rot_robot @ self.reference_ee_rot_robot

                # Transform the next pose from base frame to arm frame
                next_ee_pos = base2side.T @ (next_ee_pos_robot - base2side_pos)
                next_ee_rot = base2side.T @ next_ee_rot_robot

                rr.log(
                    f"robot/delta_pos_robot",
                    rr.LineStrips3D([[0, 0, 0], delta_pos_robot]),
                )

                rr.log(
                    f"robot/{self.which_robot}_arm/next_ee",
                    rr.Transform3D(
                        translation=next_ee_pos, mat3x3=next_ee_rot, axis_length=0.1
                    ),
                )

                next_ee_rot_euler = euler.mat2euler(next_ee_rot)
                rr.log("euler/roll", rr.Scalar(next_ee_rot_euler[0]))
                rr.log("euler/pitch", rr.Scalar(next_ee_rot_euler[1]))
                rr.log("euler/yaw", rr.Scalar(next_ee_rot_euler[2]))
                action = ActionArmAgent(
                    t=time(),
                    ee_pos=next_ee_pos,
                    ee_rot=next_ee_rot,
                    **controller_inputs,
                )
                return action
        else:
            if (
                self.control_state == ControlState.ACTIVE
            ):  # control was just deactivated
                self.on_control_deactivated()
            self.control_state = ControlState.INACTIVE
            self.activation_countdown = self.activation_countdown_max
            self.reference_vr_pose = None
            return None

    def get_oculus_data_keys_for_side(self):
        match self.which_hand:
            case "l":
                pose_key = "l"
                trigger_key = "leftTrig"
                power_grasp_key = "leftGrip"
                joystick_key = "leftJS"
            case "r":
                pose_key = "r"
                trigger_key = "rightTrig"
                power_grasp_key = "rightGrip"
                joystick_key = "rightJS"
            case _:
                raise NotImplementedError(f"Unknown hand: {self.which_hand}")
        return joystick_key, pose_key, power_grasp_key, trigger_key

    def get_transforms_for_side(self, obs: ObservationArmAgent):
        # Get the torso yaw position and use that to compute the base2torso transform.
        # This info isn't in obs because currently obs is only the obs for the current agent (e.g. left/right)
        # _, waist_positions = waist_api.getWaistJointsPosition()
        # torso_yaw = waist_positions[0]
        # TODO(2025/05/29, Dylan Colli): FIX ME, Waist API no longer returns values, just default 0 yaw
        torso_yaw = np.pi / 2
        base2torso = euler.euler2mat(0, 0, torso_yaw)

        if self.which_robot == "l":
            base2side_pos = base2torso @ torso2left_pos
            base2side = base2torso @ torso2left
        else:
            base2side_pos = base2torso @ torso2right_pos
            base2side = base2torso @ torso2right

        rr.log(
            f"robot/torso",
            rr.Transform3D(mat3x3=base2torso, axis_length=0.3),
        )

        # pos and rot in arm base frame
        ee_pos = obs["ee_pos"]
        ee_rot = obs["ee_rot"]

        # transform from arm frame into the base frame
        ee_pos_base = base2side @ ee_pos + base2side_pos
        ee_rot_base = base2side @ ee_rot

        return base2side_pos, base2side, ee_pos_base, ee_rot_base

    def get_controller_inputs(self, button_data, grip_key, joystick_key):
        js_x, js_y = self.get_js_xy(button_data, joystick_key)
        return ControllerInputs(
            power_grasp_frac=button_data[grip_key][0],
            thumb_lateral_frac=-js_x,
            thumb_grasp_frac=-js_y,
            A=button_data["A"],
            B=button_data["B"],
            X=button_data["X"],
            Y=button_data["Y"],
        )

    def get_js_xy(self, button_data, joystick_key):
        # (x, y) position of joystick, range (-1.0, 1.0)
        js_x = button_data[joystick_key][0]
        js_y = button_data[joystick_key][1]
        return js_x, js_y

    def has_valid_controller_data(self, pose_data, button_data):
        if len(pose_data) == 0 or len(button_data) == 0:
            return False, "No Data"
        if "X" not in button_data or "Y" not in button_data:
            msg = "Left controller missing. After rebooting, controller must be viewed by headset once to register."
            return False, msg
        if "A" not in button_data or "B" not in button_data:
            msg = "Right controller missing. After rebooting, controller must be viewed by headset once to register."
            return False, msg
        default_l_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0.2],
                [0, 0, 0, 1],
            ]
        )
        default_r_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.2],
                [0, 0, 0, 1],
            ]
        )

        # The quest controllers will go into a "sleep" mode if they are not moved recently, which results in a pose
        # of at or near identity. We do not want the robot to move if the controller is in this state!
        l_pose_is_identity = np.max(np.abs(pose_data["l"] - default_l_pose)) < 0.1
        r_pose_is_identity = np.max(np.abs(pose_data["r"] - default_r_pose)) < 0.1
        if self.which_hand == "l" and l_pose_is_identity:
            return False, "Pose is identity!"
        elif self.which_hand == "r" and r_pose_is_identity:
            return False, "Pose is identity!"

        return True, ""

    def get_obs(self):
        raise NotImplementedError("Implement in subclass")
