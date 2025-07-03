import os
from pathlib import Path

import numpy as np

from p2_teleop.constants import CAM_IDS
from p2_teleop.episode_recording.episode_utils import load_image_paths


def make_individual_video_ffmpeg(
    capture_times, episode_path, image_paths, output_filename
):
    # Convert the capture times to frame durations then save
    frame_durations = np.diff(capture_times)
    frame_durations = np.append(frame_durations, frame_durations[-1])
    durations_path = episode_path / "frame_durations.txt"
    with durations_path.open("w") as f:
        for img_path, duration in zip(image_paths, frame_durations):
            f.write(f"file '{img_path}'\nduration {duration}\n")
    # run ffmpeg to generate video
    # yuv420p does not refer to the resolution, but the pixel format, and is the most widely supported format.
    cmd = (
        f"ffmpeg -y -loglevel error -f concat -safe 0 -i {durations_path} "
        f"-vsync vfr -pix_fmt yuv420p {output_filename}"
    )
    retcode = os.system(cmd)
    if retcode != 0:
        print("bad!")


def make_combined_video_ffmpeg(combine_video_path, video_paths):
    input_files = " ".join([f"-i {video}" for video in video_paths.values()])
    filter_complex = ";".join(
        [f"[{i}:v]scale=-1:ih[{chr(97 + i)}]" for i in range(len(video_paths))]
    )
    hstack_inputs = "".join([f"[{chr(97 + i)}]" for i in range(len(video_paths))])
    # Combine videos using ffmpeg hstack filter with dynamically generated command
    cmd = (
        f"ffmpeg -y -loglevel error {input_files} "
        f'-filter_complex "{filter_complex};{hstack_inputs}hstack=inputs={len(video_paths)}" '
        f"-pix_fmt yuv420p {combine_video_path}"
    )
    retcode = os.system(cmd)
    if retcode != 0:
        print("bad!")


def gen_episode_vids(episode_path: Path):
    video_paths = {}
    for cam_name in CAM_IDS.keys():
        output_filename = episode_path / f"{episode_path.name}_{cam_name}.mp4"

        if output_filename.exists():
            video_paths[cam_name] = output_filename
            continue

        capture_times, image_paths = load_image_paths(cam_name, episode_path)

        if len(capture_times) < 10:
            continue

        make_individual_video_ffmpeg(
            capture_times, episode_path, image_paths, output_filename
        )
        video_paths[cam_name] = output_filename

    if len(video_paths) < 1:
        return

    combine_video_path = episode_path / f"combined_{episode_path.name}.mp4"
    if combine_video_path.exists():
        return

    make_combined_video_ffmpeg(combine_video_path, video_paths)
