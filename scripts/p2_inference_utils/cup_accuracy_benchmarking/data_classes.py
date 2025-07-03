from dataclasses import dataclass, field
from itertools import product
from typing import Optional
import numpy as np
import yaml
import rerun as rr

from p2_teleop.constants import DEFAULT_LEFT_IP
from p2_teleop.cup_accuracy_benchmarking.keyboard_yaw_control import (
    interactive_yaw_control,
)
from p2_teleop.p2.diana_api import getJointPos, getTcpPos
from p2_teleop.p2.p2_math_utils import pose_to_pos_rot
from p2_teleop.visualization.rerun_utils import log_arm_frames


def current_tcp_pose(ip=DEFAULT_LEFT_IP):
    """
    Get the current TCP pose (position and rotation) of the robot.
    """
    pos_axangle = np.zeros(6)
    getTcpPos(pos_axangle, ipAddress=ip)
    ee_pos, ee_rot = pose_to_pos_rot(pos_axangle)
    return ee_pos, ee_rot


def current_tcp_pose_axangle(ip=DEFAULT_LEFT_IP):
    """
    Get the current TCP pose of the robot in axangle format."""
    pos_axangle = np.zeros(6)
    getTcpPos(pos_axangle, ipAddress=ip)
    return np.array(pos_axangle)


@dataclass
class ApproachSettings:
    """
    Control parameters for the robot's vertical approach during pick experiments.

    :param use_vertical_approach: Enable vertical approach (True/False).
    :param approach_height: Height (in meters) to move above the object.
    :param approach_velocity: Speed during the vertical movement.
    :param approach_acceleration: Acceleration during the vertical movement.
    """

    use_vertical_approach: bool = False
    approach_height: float = 0.1
    approach_velocity: float = 0.3
    approach_acceleration: float = 0.3


@dataclass
class TestConfig:
    """
    Test configuration with sampling parameters and ranges.

    :param sampling_strategy: Strategy for sampling adjustments, e.g. "grid".
    :param x_range: A list (start, stop, step) for the x-axis adjustments (in meters).
    :param y_range: A list (start, stop, step) for the y-axis adjustments (in meters).
    :param z_range: A list (start, stop, step) for the z-axis adjustments (in meters).
    :param roll_angles: A list of roll angles (in degrees).
    :param pitch_angles: A list of pitch angles (in degrees).
    :param yaw_angles: A list of yaw angles (in degrees).
    :param axis_combinations: list defining which axes are sampled together.
    :param set_yaw_angle: The initial yaw angle setting (in degrees).
    :param approach_settings: An instance of ApproachSettings, controlling vertical approach.
    :ivar delay_between_tests_s: Delay between tests in seconds.
    """

    # Config parameters
    sampling_strategy: str
    x_range: list[float, float, float]
    y_range: list[float, float, float]
    z_range: list[float, float, float]
    roll_angles: list[float]
    pitch_angles: list[float]
    yaw_angles: list[float]
    axis_combinations: list[str]
    set_yaw_angle: float = 0.0
    approach_settings: ApproachSettings = field(default_factory=ApproachSettings)
    delay_between_tests_s: float = 0.0

    def initialize_visualization(self):
        """
        Initialize the visualization environment.

        This method initializes the rerun visualization tool and logs
        the robot's arm frames to provide a visual context for the test.

        :raises RuntimeError: If the visualization environment fails to initialize.
        """

        rr.init("Pick Testing Visualization", spawn=True)
        log_arm_frames()

    def visualize_zero_position(self):
        """
        Visualize the current zero position in the robot's TCP frame.

        Logs the zero position with a fixed axis length for clear visualization.

        :raises RuntimeError: If the zero position data is not available.
        """

        rr.log(
            "robot/l_arm/zero_pose",
            rr.Transform3D(
                translation=self.zero_pos_tcp_frame,
                mat3x3=self.zero_rot_tcp_frame,
                axis_length=0.3,
            ),
        )

    def initialize_zero_position(self):
        """
        Store both joint and pose zero positions.

        Prompts the user to move the robot to the desired zero pose, then captures and stores:
        - Joint positions as the zero_joint_pos.
        - TCP pose (translation and rotation) as zero_pose_tcp_frame and zero_rot_tcp_frame.

        :raises RuntimeError: If the joint positions cannot be acquired.
        """

        print("\nMove to desired zero position and press Enter...")
        input()

        joint_pos = np.zeros(6)
        if not getJointPos(joint_pos):
            raise RuntimeError("Failed to get joint position")
        self.zero_joint_pos = joint_pos
        print(f"Set home (joint positions): {self.zero_joint_pos}")

        # Get the zero position in TCP frame
        self.zero_pos_tcp_frame, self.zero_rot_tcp_frame = current_tcp_pose()
        self.zero_pose_tcp_frame = current_tcp_pose_axangle()

    def initialize_yaw_angle(self, hfg_agent):
        """
        Set the initial yaw angle using interactive control.

        Invokes the interactive yaw control interface and updates the set_yaw_angle.

        :param hfg_agent: The agent instance controlling the yawing gripper.
        """

        final_angle = interactive_yaw_control(hfg_agent)
        if final_angle is not None:
            print(f"\nFinal yaw angle: {final_angle:.1f} degrees")
            self.set_yaw_angle = final_angle

    def generate_test_points(self):
        """
        Generate test points based on the configured sampling strategy.

        When using the "grid" sampling strategy, iterates over the provided ranges and
        axis combinations to create a list of test adjustments.

        :returns: A list of test points with each point as a list of adjustments.
        :rtype: list[list[float]]
        """

        test_points = []

        if self.sampling_strategy == "grid":
            axis_map = {
                "x": np.arange(*self.x_range),
                "y": np.arange(*self.y_range),
                "z": np.arange(*self.z_range),
                "roll": np.arange(*self.roll_angles),
                "pitch": np.arange(*self.pitch_angles),
                "yaw": np.arange(*self.yaw_angles),
            }

            print("\nGenerated Test Points by Combination:")
            print("=====================================")

            for combo in self.axis_combinations:
                combo_points = []  # Track points for this combination
                print(f"\nTesting axes: {', '.join(combo).upper()}")

                point_arrays = [axis_map[axis.lower()] for axis in combo]
                for values in product(*point_arrays):
                    point = [0.0] * 6
                    for axis, value in zip(combo, values):
                        idx = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[
                            axis.lower()
                        ]
                        point[idx] = value
                    combo_points.append(point)
                    test_points.append(point)

                # Print points for this combination only
                for point in combo_points:
                    formatted_point = [
                        f"{val:.3f}" if val != 0 else "0.0" for val in point
                    ]
                    print(
                        f"  Point: (x:{formatted_point[0]}m, y:{formatted_point[1]}m, z:{formatted_point[2]}m, "
                        f"roll:{formatted_point[3]}deg, pitch:{formatted_point[4]}deg, yaw:{formatted_point[5]}deg)"
                    )
                print(f"Points in this combination: {len(combo_points)}")

            print(f"\nTotal test points generated: {len(test_points)}\n")
            return test_points

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Create a TestConfig instance from a YAML configuration file.

        Reads the YAML file specified by ``yaml_path`` and extracts the configuration
        from the "test_config" section. Initializes the TestConfig parameters including
        sampling strategy, positional and angular ranges, axis combinations, and
        approach settings.

        :param yaml_path: The path to the YAML configuration file.
        :type yaml_path: str
        :returns: An instance of TestConfig populated with the YAML configuration.
        :rtype: TestConfig
        """

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)["test_config"]

        approach_config = config.get("approach_settings", {})
        approach_settings = ApproachSettings(
            use_vertical_approach=approach_config.get("use_vertical_approach", False),
            approach_height=approach_config.get("approach_height", 0.1),
            approach_velocity=approach_config.get("approach_velocity", 0.3),
            approach_acceleration=approach_config.get("approach_acceleration", 0.3),
        )

        return cls(
            sampling_strategy=config["sampling_strategy"],
            x_range=config["x_range"],
            y_range=config["y_range"],
            z_range=config["z_range"],
            roll_angles=config["roll_angles"],
            pitch_angles=config["pitch_angles"],
            yaw_angles=config["yaw_angles"],
            axis_combinations=config["axis_combinations"],
            approach_settings=approach_settings,
        )


@dataclass
class TestObject:
    """
    Defines a test object for pick experiments.

    :param object_id: Unique identifier for the test object.
    :param object_type: Type of object (e.g., "box", "pencil", "dvd_case").
    :param zero_pose_axangle_global: Optional TCP pose (axangle representation) for the object.
    """

    object_id: int
    object_type: str
    zero_pose_axangle_global: Optional[np.ndarray] = None


@dataclass
class PickParameters:
    """
    Contains pick attempt parameters including positional and angular adjustments.

    :param adjustment_x_mm: adjustment along the x-axis in millimeters.
    :param adjustment_y_mm: adjustment along the y-axis in millimeters.
    :param adjustment_z_mm: adjustment along the z-axis in millimeters.
    :param roll_deg: adjustment in roll (degrees).
    :param pitch_deg: adjustment in pitch (degrees).
    :param yaw_deg: adjustment in yaw (degrees).
    :param vacuum_threshold: The threshold vacuum pressure to consider a successful grasp (psi).
    :param cup_id: Identifier of the cup to be used in the attempt.
    :param hold_duration_s: Duration to hold the object after successful grasp (seconds).
    """

    adjustment_x_mm: float
    adjustment_y_mm: float
    adjustment_z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    vacuum_threshold: float
    cup_id: int
    hold_duration_s: float = 3.0


@dataclass
class CupConfig:
    """
    Cup configuration parameters specific to a suction cup design.

    :param cup_id: Unique identifier for the cup.
    :param cup_type: The type or design of the cup.
    :param diameter_mm: The diameter of the cup in millimeters.
    :param cup_vacuum_threshold: The vacuum threshold required for a successful grasp.
    """

    cup_id: int
    cup_type: str
    diameter_mm: float
    cup_vacuum_threshold: float


@dataclass
class TestResult:
    """
    Captures the result of a single pick attempt.

    :param timestamp: The start time of the pick attempt.
    :param cup_id: The identifier for the suction cup used.
    :param object_id: The identifier for the test object.
    :param object_type: The type of object tested.
    :param adjustment_x_mm: The x-axis adjustment (mm) applied.
    :param adjustment_y_mm: The y-axis adjustment (mm) applied.
    :param adjustment_z_mm: The z-axis adjustment (mm) applied.
    :param roll_deg: The roll adjustment (degrees) applied.
    :param pitch_deg: The pitch adjustment (degrees) applied.
    :param yaw_deg: The yaw adjustment (degrees) applied.
    """

    timestamp: float
    cup_id: int
    object_id: int
    object_type: str
    adjustment_x_mm: float
    adjustment_y_mm: float
    adjustment_z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass
class PickResult:
    """
    Captures the result of a single pick attempt.

    :param grasp_success: Boolean indicating if the grasp was successful.
    :param lift_success: Boolean indicating if the lift was successful.
    :param grasp_pressure_psi: Vacuum pressure at the grasp step (PSI).
    :param lift_pressure_psi: Vacuum pressure after lift (PSI).
    """

    grasp_success: bool
    lift_success: bool
    grasp_pressure_psi: float
    lift_pressure_psi: float
