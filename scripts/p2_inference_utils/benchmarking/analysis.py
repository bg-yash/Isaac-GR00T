import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np

from p2_teleop.episode_recording.episode_utils import load_episode
from p2_teleop.episode_recording.episode import Episode
from p2_teleop.constants import TransferTermination


def episode_success(episode: Episode):
    return episode_metadata_success(episode.metadata)


def episode_metadata_success(metadata):
    if "success" in metadata:
        return metadata["success"]
    elif "transfer_termination" in metadata:
        return metadata["transfer_termination"] == TransferTermination.SUCCESS.name
    else:
        return False


def get_last_time_for_obs(state):
    """A state has two capture times, one for the left side and one for the right. This gets the latest one."""
    times = []
    for k, v in state.items():
        if "t" in v:
            times.append(v["t"])
    if len(times) == 0:
        raise ValueError("No 't' key found in state")
    return max(times)


def get_first_time_for_obs(state):
    """A state has two capture times, one for the left side and one for the right. This gets the earliest one."""
    times = []
    for k, v in state.items():
        if "t" in v:
            times.append(v["t"])
    if len(times) == 0:
        raise ValueError("No 't' key found in state")
    return min(times)


def get_episode_start_end_times(episode: Episode):
    """
    Get the last time of the first observation and the last time of the last observation in the episode.

    :param episode: Episode
    :return: start_time, end_time
    """
    start_time = episode.state_action_data[0]["time"]
    end_time = episode.state_action_data[-1]["time"]
    for times in episode.capture_times_dict.values():
        times = sorted(times)  # sort just in case...
        if times[0] > start_time:
            start_time = times[0]
        if times[-1] > end_time:
            end_time = times[-1]
    return start_time, end_time


def compute_episode_duration(episode: Episode):
    """
    returns episode duration in seconds, accounting for operator pauses when not operating robot

    :param episode: episode to compute duration of
    :type episode: Episode
    """
    # Accumulate all delta ts in timestamps (camera timestamps seem okay)
    #  - use numpy diff for dts
    #  - zero out all dt > threshold (running around 25 fps)
    #  - sum dt array
    scene_cam_times = np.array(episode.capture_times_dict["scene"])
    scene_cam_dts = np.diff(scene_cam_times)
    max_expected_dt_between_frames = (
        1 / 20
    )  # should be slightly slower than the true expected frame rate of 25-30 FPS
    scene_cam_dts[scene_cam_dts > max_expected_dt_between_frames] = 0
    duration = np.sum(scene_cam_dts)
    return duration


def make_episode_kpi_dict(bm):
    metadata = bm.episode.metadata
    episode = bm.episode

    episode_duration = compute_episode_duration(episode)

    if "tote_id" in metadata:
        tote_type = "tote"
    elif "tote_type" in metadata:
        tote_type = metadata["tote_type"]
    else:
        raise ValueError("No tote_id or tote_type found in metadata")
    if "transfer_termination" in metadata:
        success = metadata["transfer_termination"] == TransferTermination.SUCCESS.name
    elif "success" in metadata:
        success = metadata["success"]
    else:
        raise ValueError("No transfer_termination or success found in metadata")
    kpi_dict = {
        "benchmarking_datum_uuid": bm.data["uuid"],
        "product_id": metadata["product_id"],
        "tote_type": tote_type,
        "division_id": metadata["division_id"],
        "gripper_id": metadata["gripper_id"],
        "pack_type": metadata["pack_type"],
        "task_type": metadata["task_type"],
        "episode_duration": episode_duration,
        "success": success,
    }
    return kpi_dict


def episode_path_generator(benchmark_root: Path, ignore_in_progress: bool = True):
    """
    :param ignore_in_progress:
    :param benchmark_root: Normally '/data/benchmarking_data'

    :return: A generator that yields episode paths
    """
    for benchmark_data, benchmark_path in benchmark_data_generator(
        benchmark_root, ignore_in_progress
    ):
        for episode_path in benchmark_data["episode_paths"]:
            yield Path(episode_path), benchmark_data


def benchmark_data_generator(benchmark_root: Path, ignore_in_progress: bool = True):
    """
    :param ignore_in_progress:
    :param benchmark_root: Normally '/data/benchmarking_data'

    :return: A generator that yields episode paths
    """
    benchmark_paths = list(benchmark_root.rglob("*.json"))
    for benchmark_path in benchmark_paths:
        if ignore_in_progress and "in_progress" in str(benchmark_path):
            continue
        with open(benchmark_path, "r") as f:
            benchmark_data = json.load(f)
        yield benchmark_data, benchmark_path


@dataclass
class Benchmark:
    path: Path
    data: Dict
    episode: Episode


def benchmark_episode_generator(
    benchmark_root: Path,
    ignore_in_progress: bool = True,
    load_images: bool = False,
    skip_empty: bool = True,
    cams: List[str] | None = None,
):
    for benchmark_datum, benchmark_path in benchmark_data_generator(
        benchmark_root, ignore_in_progress
    ):
        for bm in episodes_for_benchmark_generator(
            benchmark_datum, benchmark_path, load_images, skip_empty, cams
        ):
            yield bm


def episodes_for_benchmark_generator(
    benchmark_datum,
    benchmark_path,
    load_images: bool = False,
    skip_empty: bool = True,
    cams: List[str] | None = None,
):
    for episode_path in benchmark_datum["episode_paths"]:
        episode_path = Path(episode_path)
        episode = load_episode(episode_path, load_images, cams=cams)
        images_missing = load_images and len(episode.images_dict) == 0
        state_action_missing = len(episode.state_action_data) == 0
        if any([images_missing, state_action_missing]) and skip_empty:
            continue

        yield Benchmark(benchmark_path, benchmark_datum, episode)
