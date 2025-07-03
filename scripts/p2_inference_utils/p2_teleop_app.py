from enum import Enum
from time import perf_counter

import numpy as np
import rerun as rr
from oculus_reader import OculusReader

from p2_teleop.agents.waist_agent import P2WaistAgent
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.agents.quest_p2_arm_hand_agent import QuestP2ArmHandAgent
from p2_teleop.agents.quest_p2_arm_hfg_agent import QuestP2ArmHFGAgent
from p2_teleop.visualization.rerun_utils import log_base_frames
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class ArmChoice(str, Enum):
    BOTH = "both"
    LEFT = "left"
    RIGHT = "right"


class ControllerChoice(str, Enum):
    L = "l"
    R = "r"


class Agents:
    def __init__(self, args, oculus_reader, event_manager):
        match args.arm:
            case ArmChoice.BOTH:
                self.left_agent = QuestP2ArmHFGAgent(
                    ip=args.left_ip,
                    hfg_valve_ip=args.valve_ip,
                    which_hand="r",
                    which_robot="l",
                    event_manager=event_manager,
                    oculus_reader=oculus_reader,
                    pos_scale=args.pos_scale,
                    min_grasp_valve_position=args.min_grasp_valve_position,
                    max_grasp_valve_position=args.max_grasp_valve_position,
                    min_grasp_valve_position_limit=args.min_grasp_valve_position_limit,
                    max_grasp_valve_position_limit=args.max_grasp_valve_position_limit,
                    yawing_gripper_reset_position=args.yawing_gripper_reset_position,
                    dry_run=args.dry_run,
                )
                self.right_agent = QuestP2ArmHandAgent(
                    ip=args.right_ip,
                    pos_scale=args.pos_scale,
                    which_hand="l",
                    which_robot="r",
                    event_manager=event_manager,
                    oculus_reader=oculus_reader,
                    dry_run=args.dry_run,
                )
            case ArmChoice.LEFT:
                self.right_agent = None
                self.left_agent = QuestP2ArmHFGAgent(
                    ip=args.left_ip,
                    hfg_valve_ip=args.valve_ip,
                    which_hand=args.controller.value,
                    which_robot="l",
                    event_manager=event_manager,
                    oculus_reader=oculus_reader,
                    pos_scale=args.pos_scale,
                    min_grasp_valve_position=args.min_grasp_valve_position,
                    max_grasp_valve_position=args.max_grasp_valve_position,
                    min_grasp_valve_position_limit=args.min_grasp_valve_position_limit,
                    max_grasp_valve_position_limit=args.max_grasp_valve_position_limit,
                    yawing_gripper_reset_position=args.yawing_gripper_reset_position,
                    dry_run=args.dry_run,
                )
            case ArmChoice.RIGHT:
                self.left_agent = None
                self.right_agent = QuestP2ArmHandAgent(
                    ip=args.right_ip,
                    pos_scale=args.pos_scale,
                    which_hand=args.controller.value,
                    which_robot="r",
                    event_manager=event_manager,
                    oculus_reader=oculus_reader,
                    dry_run=args.dry_run,
                )
            case _:
                raise NotImplementedError(f"Invalid robot arm choice: {args.arm}")

        if args.no_waist:
            self.waist_agent = None
        else:
            self.waist_agent = P2WaistAgent(
                event_manager=event_manager,
                oculus_reader=oculus_reader,
                ip=args.left_ip,
                dry_run=args.dry_run,
            )

    def as_list(self):
        return filter(lambda a: a is not None, [self.left_agent, self.right_agent])

    def as_dict(self):
        agents_dict = {
            "left": self.left_agent,
            "right": self.right_agent,
            "waist": self.waist_agent,
        }
        return agents_dict.items()


INSTRUCTIONS = """INSTRUCTIONS:

    Press and hold index finger trigger button to start moving the robot.
    Press the middle finger trigger button to close the hand/high-flow gripper.
    Release to open.
    Press the X button to switch grasp type.
    Press the Y button to switch between base frame and camera frame control on the left arm.
    Press the B button to switch between base frame and camera frame control on the right arm.
    Long-Click Both buttons (AB) or (XY) to clear errors and release the brake, so you can resume teleop.

"""


class Teleop:

    def __init__(self, args, node: "Node"):
        np.set_printoptions(suppress=True, precision=3)
        self.node = node
        # initialize rerun connection if enabled
        rr.init("teleop", init_logging=args.rerun)
        rr.connect_tcp()
        log_base_frames()

        if args.dry_run:
            print("No Quest, running in debug mode")
            self.oculus_reader = None
        else:
            print("Not running in debug")
            self.oculus_reader = OculusReader()

        self.event_manager = ButtonEventManager()

        # create "agents" to control left, right or both arms via teleop controller
        self.agents = Agents(args, self.oculus_reader, self.event_manager)

        print(INSTRUCTIONS)
        print("Ready! Starting teleop loop!")

        self.last_t = None

    def step(self):
        _, button_data = self.oculus_reader.get_transformations_and_buttons()
        if button_data is None or len(button_data) == 0:
            print("No data, VR not yet ready")

        self.event_manager.process_events(button_data)

        # calculate and apply current controller delta & actions to each robot
        actions = {}
        observations = {}
        for agent_name, agent in self.agents.as_dict():
            if agent is None:
                continue
            observation, action = agent.step()
            observations[agent_name] = observation
            actions[agent_name] = action

        if self.last_t is not None:
            period = perf_counter() - self.last_t
            freq = 1 / period
            rr.log("freq/teleop_app", rr.Scalar(freq))
        self.last_t = perf_counter()

        return observations, actions
