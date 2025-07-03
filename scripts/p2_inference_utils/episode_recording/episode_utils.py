import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import List
import sys
from itertools import compress

import cv2
from p2_teleop.constants import CAM_NAMES, LOGS_GROUP
from p2_teleop.episode_recording.episode import Episode
from p2_teleop.exceptions import MissingDataException

CAPTURE_TIMES_CSV = "capture_times.csv"
DATA_PKL = "data.pkl"
METADATA_JSON = "metadata.json"


def all_episode_paths_generator(root: Path):
    # look for the metadata.json files to indicate that the parent dir is an episode
    for episode_metadata_path in sorted(root.rglob(METADATA_JSON)):
        episode_dir = episode_metadata_path.parent
        yield episode_dir


def load_all_camera_images(
    episode_dir: Path, load_images: bool = True, cams: List[str] | None = None
):
    """

    :param episode_dir:
    :param load_images: If false, only load the capture times and return an empty dict for the images
    :param cams: List of camera names to load. If None, load all cameras.
    :return:
    """
    if cams is None:
        cams = CAM_NAMES.values()

    image_paths = {}
    images = {}
    capture_times = {}
    for subdir in episode_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in cams:
            subdir_capture_times, subdir_image_paths = load_image_paths(
                subdir.name, episode_dir
            )
            subdir_images = []
            if load_images:
                for img_path in subdir_image_paths:
                    img_rgb = load_image_rgb(episode_dir / img_path)
                    subdir_images.append(img_rgb)
            image_paths[subdir.name] = subdir_image_paths
            images[subdir.name] = subdir_images
            capture_times[subdir.name] = subdir_capture_times

    keys_to_remove = []
    # if there are no capture times, that means we didn't log anything (regardless of the value of load_images)
    for k, v in capture_times.items():
        if len(v) == 0:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del image_paths[k]
        del images[k]
        del capture_times[k]

    return image_paths, images, capture_times


def load_image_rgb(img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_episode(
    episode_path: Path, load_images: bool = False, cams: List[str] | None = None
):
    """

    :param episode_path:
    :param load_images: set to True if you need to load the images. This will be slower.
    :param cams: List of camera names to load. If None, load all cameras.
    :return:
    """
    if isinstance(episode_path, str):
        episode_path = Path(episode_path)

    state_action_data = load_episode_state_action_data(episode_path)
    metadata = load_episode_metadata(episode_path)
    image_paths_dict, images_dict, capture_times_dict = load_all_camera_images(
        episode_path, load_images=load_images, cams=cams
    )
    return Episode(
        path=episode_path,
        metadata=metadata,
        metadata_path=metadata["path"],
        image_paths_dict=image_paths_dict,
        images_dict=images_dict,
        were_images_loaded=load_images,
        capture_times_dict=capture_times_dict,
        state_action_data=state_action_data,
    )


def load_episode_state_action_data(episode_path):
    pkl_path = episode_path / DATA_PKL
    if not pkl_path.exists():
        raise MissingDataException(f"Data pkl not found {pkl_path}")
    with open(pkl_path, "rb") as f:
        state_action_data = pickle.load(f)
    return state_action_data


def load_episode_metadata(episode_path):
    metadata_path = episode_path / METADATA_JSON
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["path"] = metadata_path
    return metadata


def load_image_paths(cam_name, episode_path):
    image_paths = []
    capture_times = []
    cam_path = episode_path / cam_name
    with (cam_path / CAPTURE_TIMES_CSV).open() as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for img_filename, capture_time in csv_reader:
            full_img_filename = cam_path / img_filename
            image_paths.append(full_img_filename)
            capture_times.append(float(capture_time))
    return capture_times, image_paths


def load_first_frames(episode_path: Path):
    first_images_dict = {}
    for subdir in episode_path.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in CAM_NAMES.values():
            with (subdir / CAPTURE_TIMES_CSV).open() as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # skip header
                try:
                    first_img_filename, _ = next(csv_reader)
                except StopIteration:
                    continue
                first_img_path = subdir / first_img_filename
                img_rgb = load_image_rgb(first_img_path)
                first_images_dict[subdir.name] = img_rgb

    return first_images_dict


def trim_episode_by_timestamp(
    episode: Episode, t_start: float | None = None, t_end: float | None = None
):
    """
    Trims an episode to only include data between start_timestamp and end_timestamp

    :param episode: Episode object to trim
    :param start_timestamp: Timestamp to start trimming at
    :param end_timestamp: Timestamp to end trimming at
    :return: Trimmed episode
    """
    t_start = t_start if t_start is not None else -sys.maxsize
    t_end = t_end if t_end is not None else sys.maxsize

    # First, trim the image paths and images (if they're loaded)
    for cam_name in episode.image_paths_dict.keys():
        image_paths = episode.image_paths_dict[cam_name]
        capture_times = episode.capture_times_dict[cam_name]

        capture_times_valid = [t_start <= t <= t_end for t in capture_times]
        episode.capture_times_dict[cam_name] = list(
            compress(capture_times, capture_times_valid)
        )
        episode.image_paths_dict[cam_name] = list(
            compress(image_paths, capture_times_valid)
        )
        if episode.were_images_loaded:
            episode.images_dict[cam_name] = list(
                compress(episode.images_dict[cam_name], capture_times_valid)
            )

    episode.state_action_data = [
        step for step in episode.state_action_data if t_start <= step["time"] <= t_end
    ]
    return episode
