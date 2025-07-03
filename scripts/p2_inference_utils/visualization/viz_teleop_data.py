from pathlib import Path
from time import sleep
from typing import List, Optional

import numpy as np
import rerun as rr

from p2_teleop.episode_recording.episode import Episode
from p2_teleop.episode_recording.episode_utils import load_image_rgb, load_image_paths
from p2_teleop.p2.p2_math_utils import rotation_magnitude
from p2_teleop.visualization.rerun_utils import (
    log_base_frames,
    log_arm_frames,
    log_tote_approx,
)


def print_summary_info(episode: Episode):
    print("Metadata:")
    print(episode.metadata)
    print(f"Episode has {len(episode.state_action_data)} state-action samples")
    if not episode.were_images_loaded:
        print(f"Images were not loaded so can't print image summary info.")
        return

    for cam_name in episode.images_dict.keys():
        cam_images = episode.images_dict[cam_name]
        img_t = cam_images[0]
        print(
            f"Camera {cam_name} has shape {img_t.shape}, and {len(cam_images)} images"
        )


def log_init():
    """Required to set up for the other logging functions in this file"""
    rr.set_time_sequence("step", 0)
    log_base_frames()
    log_arm_frames()
    log_tote_approx()


def log_state_action_data(state_action_data, start_viz_t: Optional[float] = 0):
    start_capture_t = state_action_data[0]["obs"]["left"]["t"]
    end_capture_t = state_action_data[-1]["obs"]["left"]["t"]

    ee_pos_trajs = {}
    for t, step in enumerate(state_action_data):
        state_dict = step["obs"]
        for side, side_state_dict in state_dict.items():
            if side == "waist":
                log_arm_frames(torso_yaw=side_state_dict["waist_yaw"])
            else:
                state_capture_t = side_state_dict["t"]
                rr.set_time_seconds("capture_time", state_capture_t)
                state_viz_time = start_viz_t + (state_capture_t - start_capture_t)
                rr.set_time_seconds("viz_time", state_viz_time)
                rr.log(
                    f"robot/{side[0]}_arm/ee",
                    rr.Transform3D(
                        translation=side_state_dict["ee_pos"],
                        mat3x3=side_state_dict["ee_rot"],
                        axis_length=0.1,
                    ),
                )
                # EE trajs
                if side not in ee_pos_trajs:
                    ee_pos_trajs[side] = []
                ee_pos_trajs[side].append(side_state_dict["ee_pos"])
                rr.log(
                    f"robot/{side[0]}_arm/ee_pos_traj",
                    rr.LineStrips3D(ee_pos_trajs[side]),
                )

                # Joint angles and velocities.
                joint_angles = side_state_dict["q"]
                for i, angle in enumerate(joint_angles):
                    rr.log(f"robot/{side[0]}_arm/q_{i}", rr.Scalar(angle))
                joint_vels = side_state_dict["qdot"]
                for i, vel in enumerate(joint_vels):
                    rr.log(f"robot/{side[0]}_arm/qdot_{i}", rr.Scalar(vel))

                # Compute velocities
                if t > 0:
                    prev_ee_pos = state_action_data[t - 1]["obs"][side]["ee_pos"]
                    prev_ee_rot = state_action_data[t - 1]["obs"][side]["ee_rot"]
                    ee_delta_pos = side_state_dict["ee_pos"] - prev_ee_pos
                    ee_delta_rot = rotation_magnitude(
                        side_state_dict["ee_rot"] @ prev_ee_rot.T
                    )
                    dt = state_capture_t - state_action_data[t - 1]["obs"][side]["t"]
                    ee_linear_vel = np.linalg.norm(ee_delta_pos) / dt
                    ee_angular_vel = ee_delta_rot / dt
                    rr.log(f"ee_velocity/linear/{side}", rr.Scalar(ee_linear_vel))
                    rr.log(f"ee_velocity/angular/{side}", rr.Scalar(ee_angular_vel))
                    rr.log(
                        f"robot/{side[0]}_arm/ee_velocity",
                        rr.Arrows3D(origins=prev_ee_pos, vectors=ee_delta_pos),
                    )

        action_dict = step["actions"]
        for side, side_action_dict in action_dict.items():
            if side_action_dict is None:
                continue
            if side == "waist":
                pass
            else:
                action_capture_t = side_action_dict["t"]
                action_viz_time = start_viz_t + (action_capture_t - start_capture_t)
                rr.set_time_seconds("viz_time", action_viz_time)
                rr.log(
                    f"robot/{side[0]}_arm/ee_cmd",
                    rr.Transform3D(
                        translation=side_action_dict["ee_pos"],
                        mat3x3=side_action_dict["ee_rot"],
                        axis_length=0.1,
                    ),
                )

        if action_dict["left"] is not None:
            rr.log("valve_position", rr.Scalar(action_dict["left"]["valve_position"]))

    return start_viz_t + (end_capture_t - start_capture_t)


def log_camera_feeds(
    camera_images_dict, capture_times_dict, start_viz_t: Optional[float] = 0
):
    """
    Log camera feeds to rerun based on the capture times. In the viewer, the images all get logged,
    and then you have to hit play to see them displayed as if watching realtime.
    This allows you to layer different logged data on top of each other irrespective of how the order
    or how long it takes to load/log the data.

    :param camera_images_dict:
    :param capture_times_dict:
    :param start_viz_t: The time to start the visualization at. If None, will start at 0.
    :return:
    """
    for cam_name in camera_images_dict.keys():
        capture_times = capture_times_dict[cam_name]
        cam_images = camera_images_dict[cam_name]
        capture_t_prev = None
        for img_t, capture_t in zip(cam_images, capture_times):
            first_capture_t = capture_times[0]
            rr.set_time_seconds("capture_time", capture_t)
            viz_time = start_viz_t + (capture_t - first_capture_t)
            rr.set_time_seconds("viz_time", viz_time)
            assert np.all(img_t >= 0)
            assert img_t.dtype == np.uint8
            assert img_t.shape[2] == 3

            rr.log(cam_name, rr.Image(img_t))

            if capture_t_prev is None:
                freq = 0
            else:
                freq = 1 / (capture_t - capture_t_prev)
            rr.log(f"freq/{cam_name}", rr.Scalar(freq))
            capture_t_prev = capture_t

    # just use whatever the last capture_times list is
    return start_viz_t + (capture_times[-1] - capture_times[0])


def log_camera_feeds_approx_realtime(
    episode_dir: Path,
    cam_name: str,
    capture_times: List,
    image_paths: List,
    realtime_rate=1.0,
):
    """
    Attempts to play back the images in real time while loading them.
    Cannot be easily composed with other streams of data.

    :param episode_dir: Path to the episode directory
    :param cam_name:
    :param capture_times: Must be sorted! Order must match images
    :param image_paths: Order must match images
    :param realtime_rate: The rate at which to play back the images. 2.0 means 2x speed.
    :return:
    """

    # now log to rerun and sleep to approximate real time
    last_t = capture_times[0]
    for capture_t, img_path in zip(capture_times, image_paths):
        img_rgb = load_image_rgb(episode_dir / img_path)
        sleep_dt = (capture_t - last_t) / realtime_rate
        sleep(sleep_dt)
        rr.log(cam_name, rr.Image(img_rgb))

        last_t = capture_t


def load_log_cam_approx_realtime(episode_path, realtime_rate=6.0):
    """
    Load the scene camera images and log them in approximate real time.
    Cannot be easily composed with other streams of data. If you want to do that,
    use log_camera_feeds instead.

    :param episode_path:
    :return:
    """
    capture_times, image_paths = load_image_paths("scene", episode_path)
    log_camera_feeds_approx_realtime(
        episode_path,
        "scene",
        capture_times,
        image_paths,
        realtime_rate=realtime_rate,
    )
