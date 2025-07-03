import io
import json
from pprint import pprint

import numpy as np
import rerun as rr
from transforms3d import euler

from p2_teleop.episode_recording.episode import Episode
from p2_teleop.static_transforms import (
    torso2left_pos,
    torso2right_pos,
    torso2left,
    torso2right,
)


def log_base_frames(prefix=None):
    if prefix is None:
        prefix = "robot"

    rr.log(prefix, rr.Transform3D())
    rr.log(
        f"{prefix}/base_box",
        rr.Boxes3D(centers=[-0.25, 0, 0], sizes=[[1.15, 0.8, 0.001]]),
    )


def log_arm_frames(prefix=None, torso_yaw=np.deg2rad(90)):
    if prefix is None:
        prefix = "robot"

    base2torso = euler.euler2mat(0, 0, torso_yaw)
    base2left_pos = base2torso @ torso2left_pos
    base2left = base2torso @ torso2left
    base2right_pos = base2torso @ torso2right_pos
    base2right = base2torso @ torso2right

    rr.log(
        f"{prefix}/torso",
        rr.Transform3D(mat3x3=base2torso, axis_length=0.3),
    )

    rr.log(
        f"{prefix}/l_arm",
        rr.Transform3D(
            translation=base2left_pos[:3],
            mat3x3=base2left,
            axis_length=0.3,
        ),
    )
    rr.log(
        f"{prefix}/r_arm",
        rr.Transform3D(
            translation=base2right_pos[:3],
            mat3x3=base2right,
            axis_length=0.3,
        ),
    )


def log_tote_approx(prefix=None):
    if prefix is None:
        prefix = "robot"

    rr.log(
        f"{prefix}/tote_approx",
        rr.Boxes3D(
            centers=[0, 0.96, -0.33 / 2], half_sizes=[0.66 / 2, 0.45 / 2, 0.33 / 2]
        ),
    )


def log_episode_summary_info(episode: Episode):
    rr.log("episode_name", rr.TextDocument(episode.path.name))
    ss = io.StringIO()
    pprint(episode.metadata, ss)
    rr.log("metadata", rr.TextDocument(ss.getvalue()))
