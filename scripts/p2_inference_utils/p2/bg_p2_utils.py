import logging
import time
from subprocess import run, PIPE

import numpy as np
from bg_high_flow_valve.high_flow_valve_jvl import HighFlowValveJVL
from colorama import Fore
from common_industrial_protocol import cip_socket
from common_industrial_protocol.messages import MessageError

from p2_teleop.p2 import diana_api, hand_api, waist_api
from p2_teleop.p2.hand_api import HAND_JOINT_INDEX_TO_FINGER_NAME, HAND_JOINT_INDEX
from p2_teleop.constants import (
    DEFAULT_YAWING_GRIPPER_PORT,
    DEFAULT_YAWING_GRIPPER_POSITION_TOLERANCE,
    DEFAULT_WRIST_RATE,
)
from bg_yawing_gripper.yawing_gripper import (
    YawingGripper,
    AppliedMotionProductsConnectionError,
)
from bg_wrist_3_0.wrist import Wrist, ProximitySensorMode

from p2_teleop.p2.p2_math_utils import axangles_rot_error


class RobotStateException(Exception):
    """Used to generically indicate a state we can't/won't recover from"""

    pass


def throw_if_not_ok(predicate, total_t=1.0, dt=0.1, msg=""):
    """
    A simple loop that uses sleep to wait for predicate() to return true

    :param predicate: A function that returns True if the condition is met
    :param total_t: The total time to wait before throwing an exception
    :param dt:  The time to wait before checking the condition again
    :return:
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < total_t:
        if predicate():
            return
        time.sleep(dt)
    raise RobotStateException(msg)


def setup_arm(
    ip,
    tcp_offset=None,
    overwrite_limits=False,
    payload: np.ndarray | None = None,
    move_slow: bool | None = False,
):
    if not diana_api.initSrv((ip, 0, 0, 0, 0, 0)):
        raise RuntimeError(
            "Failed to initialize arm API. Check the pedant. you may need to reboot the robot, fix the IP, etc."
        )

    assert diana_api.cleanErrorInfo(ip)

    if diana_api.getRobotState(ip) not in [
        diana_api.RobotState.IDLE,
        diana_api.RobotState.RUNNING,
    ]:
        assert diana_api.releaseBrake(ip)

    if overwrite_limits:
        lower = np.deg2rad([-180, -160, -154, -360, -28, -80])
        upper = np.deg2rad([180, 160, 154, 360, 190, 80])

        # Check that we're not currently outside of these limits, otherwise this will cause the robot
        # to enter rescue mode which is very annoying.
        current_q = np.zeros(6)
        assert diana_api.getJointPos(current_q)
        for i in range(6):
            if current_q[i] <= lower[i] or current_q[i] >= upper[i]:
                lim_str = f"[{np.rad2deg(lower[i])}, {np.rad2deg(upper[i])}]!"
                raise ValueError(
                    f"Current J{i + 1}={np.rad2deg(current_q[i])} is outside of limits {lim_str}"
                )
        assert diana_api.setJointsPositionRange(lower, upper, ipAddress=ip)

    if move_slow:
        print("!" * 80)
        print("setup_arm called with move_slow=True!")
        print("Setting lower velocity and acceleration limits!")
        print("!" * 80)
        diana_api.setMaxCartTranslationVel(0.2, ip)
        diana_api.setMaxCartTranslationAcc(0.60, ip)
        diana_api.setMaxCartRotationVel(np.deg2rad(15), ip)
        diana_api.setMaxCartRotationAcc(np.deg2rad(250), ip)
        diana_api.setMaxJointsVel(np.deg2rad([34, 30, 30, 30, 36, 36]), ip)
        diana_api.setMaxJointsAcc(np.deg2rad([168, 117, 335, 500, 1200, 1500]), ip)
    else:
        diana_api.setMaxCartTranslationVel(3.0, ip)
        diana_api.setMaxCartTranslationAcc(16.8, ip)
        diana_api.setMaxCartRotationVel(np.deg2rad(300), ip)
        diana_api.setMaxCartRotationAcc(np.deg2rad(5000), ip)
        diana_api.setMaxJointsVel(np.deg2rad([340, 300, 300, 300, 360, 360]), ip)
        diana_api.setMaxJointsAcc(
            np.deg2rad([1680, 1170, 3350, 5000, 12000, 15000]), ip
        )

    if tcp_offset is not None:
        assert diana_api.setDefaultActiveTcpPose(tcp_offset, ip)
    if payload is not None:
        assert diana_api.setActiveTcpPayload(payload, ip)


def setup_waist(ip):
    assert waist_api.initWaistSrv((ip, 0, 0, 0, 0))
    assert waist_api.waistCleanError(ip)
    # NOTE: maybe should check if the waist is already enabled? it seems to fail sometimes and I don't know why
    waist_api.waistEnableServo(0, True, ip)


def is_hand_in_passthrough(ip):
    _, hand_mode = hand_api.getHandControlMode(ip=ip)
    _, finger1_mode = hand_api.getFinger1RotateControlMode(ip=ip)
    return (
        hand_mode == hand_api.HAND_MODE.IMPEDANCETRANSMISSION
        and finger1_mode == hand_api.HAND_MODE.POSITIONTRANSMISSION
    )


def setup_hand(ip):
    assert hand_api.initHandSrv((ip, 0, 0, 0, 0))
    assert hand_api.clearHandError(ip)
    time.sleep(0.5)
    assert hand_api.enableHandServo(False, ip)
    assert hand_api.enableFinger1RotateServo(True, ip)
    setup_hand_damping(ip)
    assert hand_api.enableHandServo(True, ip)
    assert hand_api.enableFinger1RotateServo(True, ip)
    hand_api.setHandControlMode(hand_api.HAND_MODE.IMPEDANCE, ip)
    hand_api.setFinger1RotateControlMode(hand_api.HAND_MODE.POSITION, ip)


def setup_hand_passthrough(ip, init=True):
    if init:
        assert hand_api.initHandSrv((ip, 0, 0, 0, 0, 0))
        time.sleep(0.05)
    assert hand_api.clearHandError(ip=ip)

    assert hand_api.enableHandServo(False, ip=ip)
    assert hand_api.enableFinger1RotateServo(False, ip=ip)
    setup_hand_damping(ip)
    assert hand_api.enableHandServo(True, ip=ip)
    assert hand_api.enableFinger1RotateServo(True, ip=ip)
    hand_in_passthrough = is_hand_in_passthrough(ip)

    if not hand_in_passthrough:
        fix_out_of_joint_fingers(ip)

        hand_api.setHandControlMode(hand_api.HAND_MODE.POSITION)
        hand_api.handEnablePassThrough(False, ip=ip)
        hand_api.setHandControlMode(hand_api.HAND_MODE.IMPEDANCETRANSMISSION)
        hand_api.setFinger1RotateControlMode(hand_api.HAND_MODE.POSITIONTRANSMISSION)
        throw_if_not_ok(lambda: is_hand_in_passthrough(ip))
    hand_api.handEnablePassThrough(True, ip=ip)
    # assert hand_api.setPerceptualSensorDataType(hand_api.TOUCH_SENSOR_DATA_TYPE.TOUCH_SENSOR_DATA_TYPE_RAW, ip=ip)


def fix_out_of_joint_fingers(ip):
    ret, lower, upper = hand_api.getHandJointsPositionRange(ip=ip)
    assert ret

    # TODO: figure out how to recover. I tried calling moveHandJoint or moveHandJointsSync but it didn't work
    while True:
        oob = False

        ret, current_q = hand_api.getHandJointsPosition(ip=ip)
        assert ret

        # Check if any joints are out of limits
        for i in range(HAND_Q_N):
            if current_q[i] < lower[i] or current_q[i] > upper[i]:
                finger, joint = HAND_JOINT_INDEX_TO_FINGER_NAME[HAND_JOINT_INDEX(i)]
                oob = True
                print(
                    Fore.YELLOW
                    + f"Finger {finger.name} joint {joint.name} is out of limits: "
                    f"{current_q[i]:.3f} not in [{lower[i]:.3f}, {upper[i]:.3f}]"
                    + Fore.RESET
                )
        if oob:
            # Set the hand to impedance mode (not passthrough) so it's easy to push the fingers back into limits
            hand_api.setHandControlMode(hand_api.HAND_MODE.IMPEDANCE, ip=ip)
            print("Please push the finger back into limits!")
        else:
            print("Out of bounds joints fixed!")
            break
        time.sleep(1)


def setup_hand_damping(ip):
    stiff_percentage = np.array([1.0] + [1.0, 1.0, 1.0] * 5)  # *5 repeats for fingers
    damp_percentage = np.array([1.0] + [1.0, 1.0, 1.0] * 5)

    # The API docs say stiffness is between 0 and 10, and damping is between 0 and 0.25
    # so rescale from 0-1 to those ranges
    stiff = stiff_percentage * 10
    damp = damp_percentage * 0.25

    # Get and check if anything actually needs to be changed because this is a slow operation
    ret, current_stiff, current_damp = hand_api.getHandJointsImpedance(ip)
    assert ret
    if np.allclose(stiff, current_stiff) and np.allclose(damp, current_damp):
        return
    else:
        hand_api.setHandJointsImpedance(stiff, damp, ip)


def setup_hfg_valve(ip, start: bool = True):
    if run(["ping", "-c", "1", ip, "-W", "0.1"], stdout=PIPE).returncode != 0:
        return None

    for _ in range(3):
        try:
            # connect to PCM valve
            valve = HighFlowValveJVL(
                ip_address=ip,
                # 1.0 = 100% torque, used on RPC Sierra cells
                open_torque=1.0,
                close_torque=1.0,
            )
            valve._logger.setLevel(logging.ERROR)
            if start:
                print("Starting valve...")
                valve.start()
            return valve
        except (MessageError, cip_socket.ConnectionError) as e:
            print(e)
            time.sleep(0.1)
    return None


def setup_yawing_gripper(
    ip: str, is_homing_on_startup: bool = False
) -> YawingGripper | None:
    """Connects and starts the yawing gripper, optionally homing to enable motor control"""
    for _ in range(3):
        try:
            yawing_gripper = YawingGripper(
                ip,
                port=DEFAULT_YAWING_GRIPPER_PORT,
                position_tolerance=DEFAULT_YAWING_GRIPPER_POSITION_TOLERANCE,
            )
            yawing_gripper.startup()
            if is_homing_on_startup:
                yawing_gripper.home()
            return yawing_gripper
        except (AppliedMotionProductsConnectionError, TimeoutError) as e:
            print(e)
            time.sleep(0.1)
    return None


def setup_wrist(iot_ip: str, modbus_ip: str):
    return Wrist(
        iot_ip,
        modbus_ip,
        rate=DEFAULT_WRIST_RATE,
        proximity_sensor_mode=ProximitySensorMode.DISPLACEMENT,
        use_cup_sensor=True,
        use_autoswap_sensors=False,
        use_vibration_sensor=False,
        disable_pressure_sensor=False,
        disable_proximity_sensor=False,
    )


def blocking_moveL(
    cmd_pose, v: float, a: float, ip: str, timeout: float, ptol=0.01, rot_tol_deg=2
):
    """
    Calls moveL and blocks until the move is complete or the timeout is reached.

    :param cmd_pose: pose [x, y, z, ax, ay, az] where rotation is axis-angle
    :param v: velocity in m/s
    :param a: acceleration in m/s^2
    :param ip: ip address of the arm
    :param timeout: timeout in seconds
    :param ptol: position tolerance in meters
    :param rot_tol_deg: rotation tolerance in degrees
    """
    t0 = time.perf_counter()
    current_pose = np.zeros(6)
    cmd_pose = np.array(cmd_pose)
    arm_ret = diana_api.moveL(
        cmd_pose,
        v=v,
        a=a,
        ipAddress=ip,
    )
    while True:
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("Blocking move timed out.")
        diana_api.getTcpPos(tcpPose=current_pose, ipAddress=ip)
        position_error = np.linalg.norm(
            np.array(cmd_pose[0:3]) - np.array(current_pose[0:3])
        )
        orientation_error = axangles_rot_error(cmd_pose[3:6], current_pose[3:6])
        if position_error < ptol and orientation_error < np.deg2rad(rot_tol_deg):
            return
        time.sleep(0.1)


def blocking_moveJ(cmd_q, v: float, a: float, ip: str, timeout: float, rot_tol_deg=2):
    """
    Calls moveL and blocks until the move is complete or the timeout is reached.

    :param cmd_q: pose [x, y, z, ax, ay, az] where rotation is axis-angle
    :param v: velocity in m/s
    :param a: acceleration in m/s^2
    :param ip: ip address of the arm
    :param timeout: timeout in seconds
    """
    t0 = time.perf_counter()
    current_q = np.zeros(6)
    arm_ret = diana_api.moveJ(
        cmd_q,
        v=v,
        a=a,
        ipAddress=ip,
    )
    while True:
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("Blocking move timed out.")
        diana_api.getJointPos(current_q, ipAddress=ip)
        max_joint_error = np.max(np.abs(np.array(cmd_q) - np.array(current_q)))
        if max_joint_error < np.deg2rad(rot_tol_deg):
            return

        time.sleep(0.1)


def blocking_torso_move(cmd_q, v: float, a: float, ip: str, timeout: float, tol=0.01):
    """Assumes a 1-dof torso like on our P2.0"""
    t0 = time.perf_counter()
    _ = waist_api.waistMoveJoint(0, cmd_q, v, a, ip)
    while True:
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("Blocking move timed out.")
        _, current_q = waist_api.getWaistJointsPosition(ip=ip)
        if np.abs(current_q[0] - cmd_q) < tol:
            return
        time.sleep(0.1)


HAND_Q_N = 16
HAND_Q_OPEN = np.deg2rad([50, 0, 2, 2] + [0, 2, 2] * 4)
