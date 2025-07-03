from p2_teleop.cup_accuracy_benchmarking.data_classes import (
    CupConfig,
    PickParameters,
    PickResult,
    TestConfig,
    TestObject,
    TestResult,
    current_tcp_pose_axangle,
)

from transforms3d import euler
from p2_teleop.p2.diana_api import moveJToPose, setDefaultActiveTcpPose
from p2_teleop.constants import DEFAULT_LEFT_IP, DEFAULT_LEFT_HFG_TOOL_OFFSET
from p2_teleop.p2.p2_math_utils import (
    mat2diana_axangle,
    pose_to_pos_rot,
    pos_rot_to_pose,
)
from p2_teleop.agents.quest_p2_arm_hfg_agent import QuestP2ArmHFGAgent
from bg_yawing_gripper.yawing_gripper import YawingGripper
from bg_wrist_3_0_msgs.states import StateIndexes
import numpy as np
import time
import csv
import os
import argparse
from pathlib import Path
import rerun as rr


"""
START
  ↓
Generate Test Points (random/grid)
  ↓
For each test point:
  → Compute adjusted pose
  → Move to approach pose
  → Enable vacuum
  → Check for seal
  → Lift and check if successful
  → Log results
  ↓
END
"""

"""
Data Flow

Zero Position → Test Points → Adjusted Poses → Pick Attempts → Results → CSV
"""

hfg_agent = QuestP2ArmHFGAgent(
    which_hand="r", which_robot="l", event_manager=None, oculus_reader=None
)


def wait_for_vacuum_seal(threshold: float, timeout_s: float):
    """Wait for vacuum pressure to reach below threshold.

    :param threshold: Target vacuum pressure in PSI
    :param timeout_s: Maximum time to wait in seconds
    :returns: Tuple of (success_status, final_pressure_reading)
    :rtype: Tuple[bool, float]
    """

    start_time = time.time()
    last_pressure_psi = None

    while (time.time() - start_time) < timeout_s:
        wrist_state = hfg_agent.wrist.query()
        grasp_pressure_psi = wrist_state[StateIndexes.PRESSURE_INDEX]["value"] * 14.5038
        last_pressure_psi = grasp_pressure_psi
        if grasp_pressure_psi < threshold:
            print(
                f"****Vacuum seal achieved with pressure reading: {grasp_pressure_psi} psi****"
            )
            return True, grasp_pressure_psi
        time.sleep(0.1)
    return False, last_pressure_psi


# This method moves the robot.
def run_pick_attempt(params: PickParameters, config: TestConfig, obj: TestObject):
    """Execute a single pick attempt with specified parameters.

    :param params: Pick attempt parameters including adjustments and thresholds
    :param config: Test configuration including approach settings
    :param obj: Test object configuration
    :returns: Result object containing grasp and lift success status and pressure readings
    :rtype: PickResult
    """

    try:

        # Define approach and retreat adjustments
        approach_adjustment = np.array([-0.1, 0, 0])
        retreat_adjustment = np.array([0.1, 0, 0])

        # 1.If you want the adjustments to take place at some vertical height above the object and then approach the object.
        # Primary reason behind this is to avoid any collison with the object while applying the adjustments. FOR TRANSLATIONAL or YAW adjustments ONLY.
        if config.approach_settings.use_vertical_approach:
            # Move to height above object
            zero_pose_axangle_global_ = obj.zero_pose_axangle_global.copy()
            ee_pos, ee_rot_mat = pose_to_pos_rot(zero_pose_axangle_global_)
            approach_adjustment_arm_frame = ee_rot_mat @ [
                0,
                0,
                config.approach_settings.approach_height,
            ]
            zero_pose_axangle_global_[0:3] += approach_adjustment_arm_frame
            moveJToPose(
                zero_pose_axangle_global_,
                v=config.approach_settings.approach_velocity,
                a=config.approach_settings.approach_acceleration,
                ipAddress=DEFAULT_LEFT_IP,
            )
            time.sleep(5)

        # Move to the home position to adjust for the adjustment
        zero_pose_axangle_global_ = obj.zero_pose_axangle_global.copy()
        moveJToPose(zero_pose_axangle_global_, v=0.3, a=0.3, ipAddress=DEFAULT_LEFT_IP)
        time.sleep(5)

        # 2. Apply adjustments to the zero pose and move to the adjusted pose
        adjustment = np.array(
            [
                params.adjustment_x_mm,
                params.adjustment_y_mm,
                params.adjustment_z_mm,
                params.roll_deg,
                params.pitch_deg,
                params.yaw_deg,
            ]
        )

        print(
            "-----------------------------------------------------------------------------"
        )
        print(f"Current adjustments: {adjustment}\n")

        current_pose_cup_frame = current_tcp_pose_axangle()
        ee_pos, ee_rot_mat = pose_to_pos_rot(current_pose_cup_frame)
        adjustment_adjusted_pos = ee_pos + ee_rot_mat @ adjustment[0:3]
        adjustment_rot_mat = euler.euler2mat(*np.deg2rad(adjustment[3:6]), "sxyz")
        adjustment_adjusted_rot = ee_rot_mat @ adjustment_rot_mat
        adjustment_adjusted_pose = pos_rot_to_pose(
            adjustment_adjusted_pos, adjustment_adjusted_rot
        )

        rr.log(
            "robot/l_arm/adjustment_adjusted_pose",
            rr.Transform3D(
                translation=adjustment_adjusted_pos,
                mat3x3=adjustment_adjusted_rot,
                axis_length=0.3,
            ),
        )

        moveJToPose(adjustment_adjusted_pose, v=0.3, a=0.3, ipAddress=DEFAULT_LEFT_IP)
        time.sleep(5)

        # 3. Approach the object
        if config.approach_settings.use_vertical_approach:
            # Move to height above object
            contact_pose_axangle = current_tcp_pose_axangle()
            ee_pos, ee_rot_mat = pose_to_pos_rot(contact_pose_axangle)
            retreat_adjustment_arm_frame = ee_rot_mat @ retreat_adjustment
            contact_pose_axangle[0:3] += retreat_adjustment_arm_frame
            moveJToPose(
                contact_pose_axangle,
                v=config.approach_settings.approach_velocity,
                a=config.approach_settings.approach_acceleration,
                ipAddress=DEFAULT_LEFT_IP,
            )
            time.sleep(5)

        # 4. Open the vacuum valve
        hfg_agent._hfg_grip(grip=True)
        print("HFG Valve: OPENED")

        # 5. Wait for vacuum seal
        grasp_status, grasp_pressure_psi = wait_for_vacuum_seal(
            params.vacuum_threshold, params.hold_duration_s
        )

        if not grasp_status:
            print("****Failed to achieve vacuum seal****")
        time.sleep(1)

        # 6. Hold the object in the air
        verify_grasp_pose_axangle = current_tcp_pose_axangle()
        ee_pos, ee_rot_mat = pose_to_pos_rot(verify_grasp_pose_axangle)
        approach_adjustment_arm_frame = ee_rot_mat @ approach_adjustment
        verify_grasp_pose_axangle[0:3] += approach_adjustment_arm_frame
        moveJToPose(verify_grasp_pose_axangle, v=0.3, a=0.3, ipAddress=DEFAULT_LEFT_IP)
        time.sleep(5)

        # 7. Check final vacuum reading to verify the grasp
        wrist_state = hfg_agent.wrist.query()
        lift_pressure_psi = wrist_state[StateIndexes.PRESSURE_INDEX]["value"] * 14.5038
        if not lift_pressure_psi < params.vacuum_threshold:
            lift_status = False
            print(
                f"****Failed to maintain vacuum seal. Pressure: {lift_pressure_psi} psi****"
            )
        else:
            lift_status = True
            print(f"****Vacuum seal maintained. Pressure: {lift_pressure_psi} psi****")

        # 8. Place the object back on the test rig
        object_drop_pose_axangle = current_tcp_pose_axangle()
        ee_pos, ee_rot_mat = pose_to_pos_rot(object_drop_pose_axangle)
        retreat_adjustment_arm_frame = ee_rot_mat @ retreat_adjustment
        object_drop_pose_axangle[0:3] += retreat_adjustment_arm_frame
        moveJToPose(object_drop_pose_axangle, v=0.3, a=0.3, ipAddress=DEFAULT_LEFT_IP)
        time.sleep(5)

        # 9. Close the vacuum valve
        hfg_agent._hfg_grip(grip=False)
        print("HFG Valve: CLOSED\n")

        return PickResult(
            grasp_success=grasp_status,
            lift_success=lift_status,
            grasp_pressure_psi=grasp_pressure_psi,
            lift_pressure_psi=lift_pressure_psi,
        )

    except Exception as e:
        print(f"Pick attempt failed: {e}")
        return False


def execute_and_log_test(
    params: PickParameters, config: TestConfig, cup: CupConfig, obj: TestObject
):
    """Execute pick test and log results.

    :param params: Pick parameters for this attempt
    :param config: Test configuration
    :param cup: Cup configuration
    :param obj: Test object configuration
    :returns: A tuple containing (test_result, pick_result)
    :rtype: Tuple[TestResult, PickResult]
    """

    pick_result = run_pick_attempt(params, config, obj)
    time.sleep(3)

    test_result = TestResult(
        timestamp=time.time(),
        cup_id=cup.cup_id,
        object_id=obj.object_id,
        object_type=obj.object_type,
        adjustment_x_mm=params.adjustment_x_mm,
        adjustment_y_mm=params.adjustment_y_mm,
        adjustment_z_mm=params.adjustment_z_mm,
        roll_deg=params.roll_deg,
        pitch_deg=params.pitch_deg,
        yaw_deg=params.yaw_deg,
    )

    return test_result, pick_result


def save_results(
    test_result: list[TestResult], pick_result: list[PickResult], output_file: str
):
    """Save test results to CSV file.

    :param test_result: list containing test result data to save
    :param pick_result: list containing pick result data to save
    :param output_file: Path to output CSV file
    :returns: None
    """

    headers = [
        "timestamp",
        "cup_id",
        "object_id",
        "object_type",
        "adjustment_x_mm",
        "adjustment_y_mm",
        "adjustment_z_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "grasp_pressure_psi",
        "lift_pressure_psi",
        "grasp_success",
        "lift_success",
    ]

    # Create file with headers if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    # Append new results
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for tr, pr in zip(test_result, pick_result):
            writer.writerow({**tr.__dict__, **pr.__dict__})


def get_cup_config_from_user():
    """Get cup configuration through interactive user input.

    :returns: Cup configuration object with user-specified parameters
    :rtype: CupConfig
    """

    print("\n=== Suction Cup Configuration ===")
    cup_id = input("Enter cup ID: ")
    cup_type = input("Enter cup type (line/side/pinch): ")
    diameter = float(input("Enter cup diameter (mm): "))
    vacuum_threshold = float(input("Enter vacuum threshold(psi): "))

    return CupConfig(
        cup_id=int(cup_id),
        cup_type=cup_type,
        diameter_mm=diameter,
        cup_vacuum_threshold=vacuum_threshold,
    )


def get_test_object_from_user():
    """Get test object configuration through interactive user input.

    :returns: Test object configuration with user-specified parameters
    :rtype: TestObject
    """

    print("\n=== Test Object Configuration ===")
    obj_id = int(input("Enter object ID: "))
    obj_type = input("Enter object type (box/pencil/dvd_case): ")

    return TestObject(
        object_id=obj_id,
        object_type=obj_type,
    )


def run_single_test(
    cup: CupConfig, obj: TestObject, config: TestConfig, output_file: str
):
    """Run complete test sequence for single cup-object pair.

    :param cup: Cup configuration
    :param obj: Test object configuration
    :param config: Test parameters and settings
    :param output_file: Path to save results
    :returns: None
    """

    # Initialize zero position
    config.initialize_yaw_angle(hfg_agent)

    # Update default TCP adjustment with the new yaw angle to transform the axes in the TCP frame
    updated_default_hfg_tool_offset = DEFAULT_LEFT_HFG_TOOL_OFFSET.copy()
    euler_rzyx = np.deg2rad(
        [0, -15, config.set_yaw_angle - np.rad2deg(YawingGripper.CENTER_POSITION)]
    )
    updated_default_hfg_tool_offset[3:6] = mat2diana_axangle(
        euler.euler2mat(*euler_rzyx, "rzyx")
    )

    # Add the height of the cup to the TCP adjustment
    updated_default_hfg_tool_offset[0] += cup.diameter_mm / 1000

    setDefaultActiveTcpPose(updated_default_hfg_tool_offset, DEFAULT_LEFT_IP)

    # Initialize and visualize zero position
    config.initialize_zero_position()
    obj.zero_pose_axangle_global = config.zero_pose_tcp_frame.copy()

    config.initialize_visualization()
    config.visualize_zero_position()
    test_points = config.generate_test_points()

    for x, y, z, roll, pitch, yaw in test_points:

        adjustments_tuple = [x, y, z, roll, pitch, yaw]

        params = PickParameters(
            adjustment_x_mm=adjustments_tuple[0],
            adjustment_y_mm=adjustments_tuple[1],
            adjustment_z_mm=adjustments_tuple[2],
            roll_deg=adjustments_tuple[3],
            pitch_deg=adjustments_tuple[4],
            yaw_deg=adjustments_tuple[5],
            vacuum_threshold=cup.cup_vacuum_threshold,
            cup_id=cup.cup_id,
        )

        test_result, pick_result = execute_and_log_test(params, config, cup, obj)
        save_results([test_result], [pick_result], output_file)

        time.sleep(config.delay_between_tests_s)


def main():
    """Main entry point for cup accuracy testing.

    Loads configuration, runs tests, and saves results.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=str,
        help="YAML file for test configuration",
        default=Path(__file__).parent / "config.yaml",
    )
    args = parser.parse_args()

    config_path = args.config_yaml

    config = TestConfig.from_yaml(config_path)

    # Single cup-object test
    cup = get_cup_config_from_user()
    obj = get_test_object_from_user()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"pick_test_{cup.cup_id}_{obj.object_id}_{timestamp}.csv"
    run_single_test(cup, obj, config, output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
