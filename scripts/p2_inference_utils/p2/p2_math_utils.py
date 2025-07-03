from enum import Enum

import numpy as np
from transforms3d import axangles, euler

from p2_teleop.static_transforms import torso2left, torso2left_pos


def pose_to_pos_rot(current_pose):
    """
    Converts the pose type returned by the Diana API to position vector and rotation matrix.

    :param current_pose: list of 6 numbers, [x, y, z, ax, ay, az] where rotation is axis-angle
    :return:
        current_pos: np.ndarray of shape (3,) with the position
        current_rot: np.ndarray of shape (3, 3)
    """
    current_pos = current_pose[:3]
    current_axis = current_pose[3:]
    current_rot = axangles.axangle2mat(current_axis, np.linalg.norm(current_axis))
    return np.array(current_pos), current_rot


def pos_rot_to_pose(pos: np.ndarray, rot: np.ndarray):
    """
    Converts the position vector and rotation matrix to the pose type used by the Diana API.
    :param pos: np.ndarray of shape (3,) with the position
    :param rot: np.ndarray of shape (3, 3)
    :return:
        current_pose: list of 6 numbers, [x, y, z, ax, ay, az] where rotation is axis-angle
    """
    axis, angle = axangles.mat2axangle(rot)
    current_axis = axis * angle
    current_pose = np.concatenate([pos, current_axis])
    return current_pose


def interp(a, b, x, xmin=0.0, xmax=1.0):
    """
    Linearly interpolate x from [xmin, xmax] to [a, b]
    """
    return a + (b - a) * (x - xmin) / (xmax - xmin)


def mat2diana_axangle(cmd_ee_rot):
    cmd_ee_axis_norm, cmd_ee_angle = axangles.mat2axangle(cmd_ee_rot)
    cmd_ee_axis = cmd_ee_axis_norm * cmd_ee_angle
    return cmd_ee_axis


def axangle2mat(axis_unnormalized):
    angle = np.linalg.norm(axis_unnormalized)
    axis = axis_unnormalized / angle
    return axangles.axangle2mat(axis, angle)


class AngleType(Enum):
    EULER = "euler"
    QUATERNION = "quaternion"
    AXANGLE = "axangle"

    def __str__(self):
        return self.value


def absolute_action2diana_pose(action_vec, angle_type: AngleType):
    """
    Converts a 6D action vector used by Octo to the 6D pose used by the robot.

    :param action_vec: [x, y, z, ax, ay, az] where rotation is euler angles
    :param angle_type: AngleType
    :return:
        ee_pose: [x, y, z, ax, ay, az] where rotation is axis-angle

    """
    action_rot = action_vec[3:6]
    match angle_type:
        case AngleType.EULER:
            axis, angle = euler.euler2axangle(*action_rot)
            axangle = axis * angle
            rot_mat = euler.euler2mat(*action_rot)
        case AngleType.AXANGLE:
            axangle = action_rot
            rot_mat = axangle2mat(axangle)
        case _:
            raise NotImplementedError(f"Angle type {angle_type} not supported")

    ee_pose = [
        action_vec[0],
        action_vec[1],
        action_vec[2],
        axangle[0],
        axangle[1],
        axangle[2],
    ]
    return ee_pose, action_vec[0:3], rot_mat


def axangles_rot_error(axangle1, axangel2):
    mat1 = axangles.axangle2mat(axangle1, np.linalg.norm(axangle1))
    mat2 = axangles.axangle2mat(axangel2, np.linalg.norm(axangel2))
    # This is the angle between the two rotation matrices
    rot_error = np.arccos((np.trace(mat1 @ mat2.T) - 1) / 2)
    return rot_error


def rotation_magnitude(rot):
    """
    Returns the magnitude of the rotation matrix. Can be used to check if a rotation is "large" or "small".
    Result is in radians, and should be within the range [0, pi].
    """
    return np.arccos((np.trace(rot) - 1) / 2)


def invert_homogeneous_matrix(T: np.ndarray):
    """
    Inverts a 4x4 homogeneous transformation matrix.
    :param T: 4x4 numpy array
    :return: 4x4 numpy array
    """
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def convert_pose_to_torso_frame(ee_pos, ee_rot):
    """construct pose in arm frame to torso frame"""
    T_torso2left = np.eye(4)
    T_torso2left[:3, :3] = torso2left
    T_torso2left[:3, 3] = torso2left_pos
    T_left2ee = np.eye(4)
    T_left2ee[:3, :3] = ee_rot
    T_left2ee[:3, 3] = ee_pos
    T_torso2ee = T_torso2left @ T_left2ee
    return T_torso2ee


def convert_torso_frame_to_left_arm_frame(T_torso2ee):
    """construct pose in torso frame to arm frame
    TODO: Consider refactoring the `ArmChoice` enum from p2_teleop_app.py and using that to create a
    general torso to arm frame function.
    """
    T_torso2left = np.eye(4)
    T_torso2left[:3, :3] = torso2left
    T_torso2left[:3, 3] = torso2left_pos
    T_left2ee = invert_homogeneous_matrix(T_torso2left) @ T_torso2ee
    return T_left2ee
