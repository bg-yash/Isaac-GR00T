import numpy as np
from transforms3d import euler

# TODO: add a 90deg yaw offset here!
vr2base = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)

# The transforms from the torso base frame to the left and right arm base frames
# TODO: these could be computed from URDF, but to make it fully general it would require parsing and FK,
#  which means we should also think about whether we want collision checking... Curobo? MoveIt? Something else?
torso2left = euler.euler2mat(-1.9003, 0.72678, -0.47536)
torso2right = euler.euler2mat(-1.9003, -0.72678, -2.6662)
# NOTE: these numbers are likely inaccurate, because the URDF only tells me where the link origin is,
#  which may or may not be the same frame as the robot uses to compute TCP pose.
torso2left_pos = np.array([0.35983, 0.087716, 0.61])
torso2right_pos = np.array([0.35983, -0.087716, 0.61])

# These depend on how the camera is mounted relative to the end-effector
cam2ee_right = euler.euler2mat(*np.deg2rad([12, 0, 90]), axes="sxyz")
cam2ee_left = euler.euler2mat(*np.deg2rad([0, 52, 90]), axes="rxyz")

# When you're facing the VR headset, +X is to the right and Z is out,
# which matches what the camera frame convention we use.
# vr2cam = np.eye(3)

# Here I'm trying out a different mapping where up and down is camera Z, instead of forward/backward.
# This is more similar to the base-frame EE control, so switching might be less confusing.
vr2cam = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)
