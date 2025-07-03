import enum
from pathlib import Path

import numpy as np

DEFAULT_SERVO_DT = 0.01
MAX_FINGER_DQ = np.deg2rad(15)
DEFAULT_LEFT_HFG_TOOL_OFFSET = np.array([0.360, 0, 0.232, 0, np.deg2rad(-15), 0])
DEFAULT_LEFT_HFG_PAYLOAD = np.array([2.5, 0.10, 0, 0.08, 0, 0, 0, 0, 0, 0])
# NOTE: unfortunately these are not stable. If you unplug/replug the cameras it may change :( :( :(
LEFT_WRIST_CAM_ID = "/dev/v4l/by-path/pci-0000:8c:00.0-usb-0:5.4:1.0-video-index0"
RIGHT_WRIST_CAM_ID = "/dev/v4l/by-path/pci-0000:e4:00.4-usb-0:2.4:1.0-video-index0"
SCENE_CAM_ID = "/dev/v4l/by-path/pci-0000:03:00.4-usb-0:1:1.0-video-index0"
CAM_IDS = {
    "scene": SCENE_CAM_ID,
    "left": LEFT_WRIST_CAM_ID,
    "right": RIGHT_WRIST_CAM_ID,
}
CAM_NAMES = {cam_id: name for name, cam_id in CAM_IDS.items()}
DEFAULT_LEFT_IP = "192.168.10.30"
DEFAULT_RIGHT_IP = "192.168.10.31"
DEFAULT_VALVE_IP = "192.168.10.10"
DEFAULT_YAWING_GRIPPER_IP = "192.168.10.11"
DEFAULT_YAWING_GRIPPER_PORT = 4000
DEFAULT_YAWING_GRIPPER_POSITION_TOLERANCE = 0.0349
DEFAULT_YAWING_GRIPPER_RESET_POSITION = 2.393
DEFAULT_WRIST_IOT_IP = "192.168.10.12"
DEFAULT_WRIST_MODBUS_IP = "192.168.10.13"
DEFAULT_WRIST_RATE = 100  # Hz
LEFT_START_CONFIG = np.deg2rad([-53, 28, 98, 22, 84, 72])
# Right arm out of the way, what's used for policy.
RIGHT_START_CONFIG = np.deg2rad([117, -43, -50, 93, 107, 57])
# Right arm in front of robot, ready to manipulate.
# RIGHT_START_CONFIG = np.deg2rad([55, -54, -88, 89, 106, 67])
# Any joystick displacement that is less than this fraction of the joystick range will be considered
# zero.
JOYSTICK_DEADZONE_FRAC = 0.1

CURRENT_VERSION_OBS_HFG = "HFG-1.1"
CURRENT_VERSION_OBS_ARM_AGENT = "base-3.0"
CURRENT_VERSION_OBS_HAND = "Hand-1.0"

P2_TELEOP_DIR = Path(__file__).resolve().parent
UI_ROOT = P2_TELEOP_DIR.parent / "resource"
CAPTURE_SETTINGS_DIR = P2_TELEOP_DIR / "qt_ui" / "capture_settings"
LOGS_GROUP = "logs"

EPISODE_TIMEOUT_SECONDS = 20

CAMERA_CONFIGS = [
    {
        "cam_name": "base",
        "start_service": "/onboard_sensors/base/start_svo_rec",
        "stop_service": "/onboard_sensors/base/stop_svo_rec",
        "camera_type": "zed2i",
    },
    {
        "cam_name": "torso",
        "start_service": "/onboard_sensors/torso/start_svo_rec",
        "stop_service": "/onboard_sensors/torso/stop_svo_rec",
        "camera_type": "zed2i",
    },
    {
        "cam_name": "robotiq",
        "start_service": "/onboard_sensors/robotiq/start_svo_rec",
        "stop_service": "/onboard_sensors/robotiq/stop_svo_rec",
        "camera_type": "zedxm",
    },
    {
        "cam_name": "hfg_left",
        "start_service": "/onboard_sensors/hfg_left/start_svo_rec",
        "stop_service": "/onboard_sensors/hfg_left/stop_svo_rec",
        "camera_type": "zedxm",
    },
    {
        "cam_name": "hfg_right",
        "start_service": "/onboard_sensors/hfg_right/start_svo_rec",
        "stop_service": "/onboard_sensors/hfg_right/stop_svo_rec",
        "camera_type": "zedxm",
    },
]


@enum.unique
class TransferTermination(enum.IntEnum):
    """
    The termination conditions of a transfer request.
    Copied form bg_core since we are not using anything else from bg_core.
    Before you add a new one, check the OldRPCEnumsThatMayOneDayBeUseful enum
    """

    # New codes start at 1000 to avoid clashing with RPC codes
    SUCCESS = 1000
    ITEMS_TOO_DEEP = 1001
    NO_FEASIBLE_GRASPS_ITEM_MOVED = 1002
    ITEM_FELL_OUT_OF_BAG = 1003
    BOX_BROKE = 1004
    ITEM_DAMAGED = 1005
    ITEM_FELL_OUTSIDE_CONTAINERS = 1006
    ITEM_OPENED = 1008  # for books, boxes with lids, pot/pans, etc.
    OPERATOR_ERROR = 1009
    BAD_QUEST_TRACKING = 1010

    # SKU is deemed incompatible with the system
    ITEM_INCOMPATIBLE = 160
    TOO_MANY_ITEMS_TRANSFERRED = 169  # If quantity_transferred > quantity_requested

    PLACE_CONTAINER_OVERFULL = 6
    PLACE_CONTAINER_OVERFULL_WITH_POTENTIAL_JAM = 7
    PLACE_CONTAINER_INSUFFICIENT_SPACE = 12

    CANCELLED = 101
    EMERGENCY_STOP = 106
    ROBOT_CRASH = 111  # robot driver went down, e.g. due to collision
    ROBOT_TORQUE_FAULT = 114  # Robot torque-faulted
    ERR_ACC_LIMIT = 115  # Robot acceleration limit exceeded

    NO_TOOLTIP_ON_GRIPPER = 110  # no tooltip (e.g. cup) on gripper

    ITEM_WRONG_PICK_DIVISION = (
        124  # used if you accidentally grab the wrong item and want to end the episode
    )

    DEBUGGING = -2
    UNKNOWN = -1


class OldRPCEnumsThatMayOneDayBeUseful(enum.IntEnum):
    """
    The below codes are very unlikely to be used... should we delete them or hide them from the combobox?
    Otherwise, it might be overwhelming.
    """

    HARDWARE_FAILURE = 108

    PICK_CONTAINER_OVERFULL = 8  # PickBinOverfullnessException exception
    PICK_CONTAINER_OVERFULL_WITH_POTENTIAL_JAM = (
        9  # PickBinOverfullnessWithJamException exception
    )
    # Yawing gripper exceptions
    YAWING_GRIPPER_HARDWARE_FAILURE = 154

    PRODUCT_UNKNOWN = 105
    BARCODE_OF_PICKED_ITEM_NOT_IN_DATABASE = 219

    # Timed out waiting for readings from a scale. This may need to be split up
    # into locale-specific versions.
    SCALE_READING_TIMEOUT = 162

    ITEM_WRONG_PLACE_DIVISION = 125
    NONITEM_WRONG_PLACE_DIVISION = 209

    MONGO_CONNECTION_FAILURE = 163
    # Blower breaker tripped
    BLOWER_BREAKER_TRIPPED = 164
    # Blower overtemp
    BLOWER_OVERTEMP = 165
    # Blower fault
    BLOWER_FAULT = 166

    # Error code for Overcurrent warnings - for now this is specific to FANUC robots.
    OVERCURRENT_IMMINENT = 167
    # Error code for an Overcurrent alarm.
    OVERCURRENT_FAILURE = 168


@enum.unique
class ToteTermination(enum.IntEnum):
    SUCCESS = 0
    GRIPPER_SKU_COMBO_IMPOSSIBLE = 1
    ITEM_TOO_DEEP = 2
    ITEM_INCOMPATIBLE = 3

    DEBUGGING = -2
    UNKNOWN = -1
