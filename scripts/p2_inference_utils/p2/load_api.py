import os
import platform
import sys
from ctypes import CDLL
from pathlib import Path


def load_api(api_name: str):
    library_names = {"Linux": f"lib{api_name}.so", "Windows": f"{api_name}.dll"}
    # /opt/Agile is a symlink to the latest version of the Agile library.
    # This is the setup we are using on the p2-dev-1 Exxact machine.
    library_dir = Path("/home/yash.shukla@berkshiregrey.com/dorkspaces/rad_p2_ws/src/bg_p2/p2_ros2/p2_hardware/p2_api/lib/")
    library_path = library_dir / library_names[platform.system()]
    assert library_path.exists()
    try:
        if sys.version_info < (3, 8):
            api_mod = CDLL(library_path)
        else:
            api_mod = CDLL(library_path, winmode=0)
    except OSError as e:
        print(e)
        print(
            "If you see an error about failing to open other .so libraries, like libToolSdk.so,"
        )
        print(
            "you need to add the path to the directory containing these .so files to LD_LIBRARY_PATH."
        )
        raise e
    return api_mod
