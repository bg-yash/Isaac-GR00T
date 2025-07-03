from typing import Dict, Any
import time

import numpy as np
# import rerun as rr

from p2_teleop.agents.quest_p2_arm_agent import QuestP2ArmAgent
from p2_teleop.constants import (
    DEFAULT_LEFT_HFG_TOOL_OFFSET,
    DEFAULT_LEFT_HFG_PAYLOAD,
    DEFAULT_SERVO_DT,
    DEFAULT_YAWING_GRIPPER_IP,
    JOYSTICK_DEADZONE_FRAC,
    DEFAULT_WRIST_IOT_IP,
    DEFAULT_WRIST_MODBUS_IP,
    DEFAULT_YAWING_GRIPPER_RESET_POSITION,
    DEFAULT_VALVE_IP,
    DEFAULT_LEFT_IP,
    CURRENT_VERSION_OBS_HFG,
)
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.p2.bg_p2_utils import setup_hfg_valve, setup_yawing_gripper, setup_wrist
from p2_teleop.p2.p2_math_utils import interp
from bg_yawing_gripper.yawing_gripper import YawingGripper
from bg_wrist_3_0.wrist import Wrist
from bg_wrist_3_0_msgs.msg import States
from p2_teleop.observations import ObservationLeft
from p2_teleop.actions import ActionLeft

# NOTE: These values were added by Dylan to simplify teleop control, allowing for discrete
# velocity command zones.
# The yawing motor can move fairly fast. We're limiting it for now.
YAW_CMD_VEL_MAX = YawingGripper.CMD_VELOCITY / 3.0
YAW_CMD_VEL_MED_FRAC = 2 / 3
YAW_CMD_VEL_MED = YAW_CMD_VEL_MAX * YAW_CMD_VEL_MED_FRAC
YAW_CMD_VEL_SLOW_FRAC = 1 / 3
YAW_CMD_VEL_SLOW = YAW_CMD_VEL_MAX * YAW_CMD_VEL_SLOW_FRAC


class QuestP2ArmHFGAgent(QuestP2ArmAgent):
    """
    Class representing P2 arm w/ BG HFG
    """

    def __init__(
        self,
        which_hand: str,
        which_robot: str,
        event_manager: ButtonEventManager | None,
        oculus_reader,  # don't default construct, since this happens at class definition time
        servo_dt=DEFAULT_SERVO_DT,
        ip: str = DEFAULT_LEFT_IP,
        hfg_valve_ip: str | None = DEFAULT_VALVE_IP,
        yawing_gripper_ip: str | None = DEFAULT_YAWING_GRIPPER_IP,
        wrist_iot_ip: str | None = DEFAULT_WRIST_IOT_IP,
        wrist_modbus_ip: str | None = DEFAULT_WRIST_MODBUS_IP,
        verbose: bool = False,
        pos_scale: np.ndarray | None = None,
        min_grasp_valve_position: float = 0.5,
        max_grasp_valve_position: float = 0.1,
        min_grasp_valve_position_limit: float = 1.0,
        max_grasp_valve_position_limit: float = 0.0,
        yawing_gripper_reset_position: float = DEFAULT_YAWING_GRIPPER_RESET_POSITION,
        start: bool = True,
        dry_run: bool = False,
    ):
        """
        :param which_hand: which hand to control
        :param which_robot: which robot to control
        :param event_manager: event manager to use
        :param oculus_reader: None will default to a new OculusReader
        :param servo_dt: time step for servoing
        :param ip: IP address of the robot
        :param hfg_valve_ip: IP address of the HFG valve
        :param yawing_gripper_ip: IP address of the yawing gripper. Give None for no yawing gripper.
        :param wrist_iot_ip: IP address of the wrist IOT. Give None for no wrist.
        :param wrist_modbus_ip: IP address of the wrist modbus. Give None for no wrist.
        :param verbose: If True, print debug messages.
        :param min_grasp_valve_position: the minimum position of the HFG valve to use when lightly
            pressing the quest trigger.
        :param max_grasp_valve_position: the maximum position of the HFG valve to use when almost
            fully pressing the quest trigger.
        :param min_grasp_valve_position_limit: The absolute minimum valve position. Must be >=
            min_grasp_valve_position.
        :param max_grasp_valve_position_limit: The absolute maximum valve position. Must be <=
            max_grasp_valve_position.
        :param yawing_gripper_reset_position: The position to reset the yawing gripper to.
        :param start: If True, the agent will start the valve.
        :param dry_run: If True, the agent will not actually send commands to the robot.
        """
        self.yawing_gripper_reset_position = yawing_gripper_reset_position
        if pos_scale is None:
            pos_scale = np.array([1.25, 1.25, 2.5])
        super().__init__(
            ip=ip,
            which_hand=which_hand,
            which_robot=which_robot,
            pos_scale=pos_scale,
            event_manager=event_manager,
            oculus_reader=oculus_reader,
            servo_dt=servo_dt,
            verbose=verbose,
            tcp_offset=DEFAULT_LEFT_HFG_TOOL_OFFSET,
            payload=DEFAULT_LEFT_HFG_PAYLOAD,
            dry_run=dry_run,
        )

        self.new_hand_cmd = False
        self.last_is_grasp = False
        self.hfg_valve_ip = hfg_valve_ip
        self.yawing_gripper_ip = (
            yawing_gripper_ip  # Can be none if not using yawing motor.
        )
        self.wrist_iot_ip = wrist_iot_ip  # Can be none if not collecting wrist data.
        self.wrist_modbus_ip = (
            wrist_modbus_ip  # Can be none if not collecting wrist data.
        )
        self._min_grasp_valve_position = min_grasp_valve_position
        self._max_grasp_valve_position = max_grasp_valve_position
        self._min_grasp_valve_position_limit = min_grasp_valve_position_limit
        self._max_grasp_valve_position_limit = max_grasp_valve_position_limit

        if not dry_run:
            self.valve = setup_hfg_valve(self.hfg_valve_ip, start=start)
        else:
            self.valve = None

        if self.valve is None:
            print(f"Valve is not connected -- is it powered off?")

        self.yawing_gripper: YawingGripper | None = None
        if self.yawing_gripper_ip is not None and not self.dry_run:
            self.yawing_gripper = setup_yawing_gripper(self.yawing_gripper_ip)
            if not self.yawing_gripper._drive.homed:
                print(
                    'WARNING: Yawing gripper has not been homed! Click the "Home Yawing Gripper" '
                    "button or program will crash when attempting to yaw the gripper."
                )

        self.wrist = None
        if not self.dry_run:
            self.wrist = self._setup_wrist()

    def _setup_wrist(self) -> Wrist | None:
        wrist = None
        if self.wrist_iot_ip is None and self.wrist_modbus_ip is None:
            print("No wrist IP provided, not setting up wrist.")
        elif self.wrist_iot_ip is not None and self.wrist_modbus_ip is not None:
            wrist = setup_wrist(self.wrist_iot_ip, self.wrist_modbus_ip)
        else:
            print("Only one wrist IP provided, not setting up wrist.")
        return wrist

    def _hfg_grip(self, grip: bool):
        """
        Actuates valve to grip or release with HFG.
        :param bool grip: Grips with HFG if True, releases otherwise.
        """
        if grip:
            # close the valve to grip
            self.valve.command_valve(False)
        else:
            # open the valve to release
            self.valve.command_valve(True)

    def on_control_deactivated(self):
        if self.yawing_gripper is not None:
            self._rotate_stop_if_moving()

    def _rotate_stop_if_moving(self):
        """
        Stops the motor if it is moving and not already stopping
        Helpful for not spamming the rotate_stop command when teleoping.
        """
        state = self.yawing_gripper.get_state()
        if state.status.moving and not state.status.stopping:
            self.yawing_gripper.rotate_stop()

    def plan(self, obs):
        action = super().plan(obs)

        if action is None:
            return action

        # Since `ActionLeft` is a subclass of the action's `ActionArmAgent` class and the `dict`
        # class, it's fair to basically cast this action to `ActionLeft` and modify it. This doesn't
        # accomplish anything functionally, but it makes type checkers happy.
        action: ActionLeft
        self._plan_valve(action)
        self._plan_yawing_gripper(action)

        return action

    def _plan_valve(self, action: ActionLeft):
        """
        Plans the HFG valve's action

        :param action: dictionary describing the buttons pressed and actions that are planned.
        :type action: ActionLeft
        """
        if self.valve is None:
            return

        grasp_frac = action["power_grasp_frac"]
        rr.log("gripper/grasp_frac", rr.Scalar(grasp_frac))
        # fully released
        if grasp_frac <= 0.01:
            valve_position = self._min_grasp_valve_position_limit
        # fully pressed
        elif grasp_frac >= 0.99:
            valve_position = self._max_grasp_valve_position_limit
        else:
            # normalize trigger state to cleanly interpolate between our min and max valve positions
            # in the non-trigger-deadzone range [0.01, 0.99].
            trigger_normalized = (grasp_frac - 0.01) / 0.98
            valve_position = interp(
                self._min_grasp_valve_position,
                self._max_grasp_valve_position,
                trigger_normalized,
            )

        action["valve_position"] = valve_position
        rr.log("gripper/valve_position", rr.Scalar(valve_position))

    def _joystick_lateral_frac_to_yawing_gripper_vel(self, lat_frac: float) -> float:
        """
        Converts the joystick's lateral fraction to the yawing gripper's velocity using zones.

        Velocity "zones" are used instead of smooth scaling as, according to the yawing motor
        driver, setting the velocity and acceleration is an expensive operation. By using zones, the
        velocity setter ignores the command if the velocity is already set to the desired value.

        :param lat_frac: The joystick's lateral fraction.
        :type lat_frac: float
        :return: The yawing gripper's velocity.
        :rtype: float
        """
        lat_frac_abs = abs(lat_frac)
        if lat_frac_abs <= JOYSTICK_DEADZONE_FRAC:
            # This function shouldn't be called if the joystick is in the "deadzone". Nevertheless,
            # just return 0 velocity in case it is in the future.
            return 0.0
        elif lat_frac_abs <= YAW_CMD_VEL_SLOW_FRAC:
            return YAW_CMD_VEL_SLOW
        elif lat_frac_abs <= YAW_CMD_VEL_MED_FRAC:
            return YAW_CMD_VEL_MED
        else:
            return YAW_CMD_VEL_MAX

    def _calc_yawing_gripper_goal_yaw(self, lat_frac: float) -> float:
        """
        Calculates the yawing gripper's goal yaw based on the joystick's lateral fraction.

        :param lat_frac: The joystick's lateral fraction.
        :type lat_frac: float
        :return: The yawing gripper's goal yaw.
        :rtype: float
        """
        lat_frac_sign = np.sign(lat_frac)
        goal_yaw = (
            self.yawing_gripper.POSITION_LIMITS[0]
            if lat_frac_sign < 0
            else self.yawing_gripper.POSITION_LIMITS[1]
        )
        return goal_yaw

    def _plan_yawing_gripper(self, action: ActionLeft):
        if self.yawing_gripper is None:
            return

        lat_frac = action["thumb_lateral_frac"]

        lat_frac_abs = abs(lat_frac)
        if lat_frac_abs <= JOYSTICK_DEADZONE_FRAC:
            action["hfg_yaw_stop_rotate"] = True
            action["hfg_yaw_cmd_vel"] = 0.0
            action["hfg_yaw_goal_yaw"] = self.yawing_gripper.get_state().position
        else:
            action["hfg_yaw_stop_rotate"] = False

            action["hfg_yaw_cmd_vel"] = (
                self._joystick_lateral_frac_to_yawing_gripper_vel(lat_frac)
            )
            action["hfg_yaw_goal_yaw"] = self._calc_yawing_gripper_goal_yaw(lat_frac)

        t_start = time.perf_counter()
        wrist_state = self.wrist.query()
        rr.log(
            "wrist/wrist_pressure_bar",
            rr.Scalar(wrist_state[States.PRESSURE_INDEX]["value"]),
        )
        rr.log(
            "wrist/wrist_gripper_proximity",
            rr.Scalar(wrist_state[States.CONTACT_INDEX]["value"]),
        )
        rr.log(
            "wrist/wrist_cup_sensor",
            rr.Scalar(wrist_state[States.CUP_SENSOR_INDEX]["value"]),
        )
        rr.log("wrist/wrist_query_time", rr.Scalar(time.perf_counter() - t_start))

    def _actuate_gripper(self, action: ActionLeft):
        if self.valve is not None:
            self.valve.command_position(
                position=action["valve_position"], slew_rate_scale=1.0
            )
        if self.yawing_gripper is not None:
            if action["hfg_yaw_stop_rotate"]:
                self._rotate_stop_if_moving()
            else:
                self.yawing_gripper.rotate(
                    action["hfg_yaw_goal_yaw"],
                    action["hfg_yaw_cmd_vel"],
                    blocking=False,
                )

    def get_episode_metadata(self):
        episode_metadata = super().get_episode_metadata()
        episode_metadata["yawing_gripper_reset_position"] = (
            self.yawing_gripper_reset_position
        )
        episode_metadata["min_grasp_valve_position"] = self._min_grasp_valve_position
        episode_metadata["max_grasp_valve_position"] = self._max_grasp_valve_position
        return episode_metadata

    def get_obs(self) -> ObservationLeft:
        obs: ObservationLeft = super().get_obs()
        obs["version"] += CURRENT_VERSION_OBS_HFG

        if self.yawing_gripper is not None:
            state = self.yawing_gripper.get_state()
            obs["hfg_yaw_position"] = state.position
            obs["hfg_yaw_cmd_position"] = state.cmd_position
            obs["hfg_yaw_cmd_velocity"] = state.cmd_velocity
            obs["hfg_yaw_cmd_acceleration"] = state.cmd_acceleration
            obs["hfg_yaw_current"] = state.current
            obs["hfg_yaw_temperature"] = state.temperature

            rr.log("hfg_yaw_position", rr.Scalar(state.position))
        if self.wrist is not None:
            wrist_state = self.wrist.query()

            obs["wrist_pressure_bar"] = wrist_state[States.PRESSURE_INDEX]
            obs["wrist_gripper_proximity"] = wrist_state[States.CONTACT_INDEX]
            obs["wrist_cup_sensor"] = wrist_state[States.CUP_SENSOR_INDEX]

        return obs

    def reset_yawing_motor_pose(self):
        """
        Resets the yawing motor to the 'reset' position (potentially different than home position)
        """
        if self.yawing_gripper:
            self.yawing_gripper.rotate(self.yawing_gripper_reset_position)
        else:
            print(
                "Warning: No yawing gripper found! Can't reset a non-existent yawing gripper!"
            )

    def home_yawing_gripper(self):
        """
        Re-homes the yawing gripper
        """
        if self.yawing_gripper:
            self.yawing_gripper.home()
        else:
            print(
                "Warning: No yawing gripper found! Can't home a non-existent yawing gripper!"
            )

    def on_reset(self, _):
        super().on_reset(_)
        if self.valve is not None and not self.valve._drive.is_connected():
            self.valve.start()
            self.valve._drive.connect()

    def close(self):
        if self.wrist:
            self.wrist.close()
        if self.yawing_gripper:
            self.yawing_gripper.shutdown()
        if self.valve:
            self.valve.close()
