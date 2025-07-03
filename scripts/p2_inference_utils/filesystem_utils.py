import shutil
from pathlib import Path
from p2_teleop.constants import LOGS_GROUP
import grp


def group_exists(group_name):
    try:
        grp.getgrnam(group_name)
        return True
    except KeyError:
        return False


def mkdir_as_logs_group(video_root: Path):
    """Make a direction, and any needed parent directories. Then chown to the logs group."""
    dirs_to_make = []
    while video_root != Path("/"):
        if video_root.exists():
            break
        dirs_to_make.append(video_root)
        video_root = video_root.parent

    for dir_to_make in reversed(dirs_to_make):
        dir_to_make.mkdir()
        if group_exists(LOGS_GROUP):
            shutil.chown(dir_to_make, group=LOGS_GROUP)
