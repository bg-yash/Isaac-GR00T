from threading import Thread, Lock
from time import perf_counter, sleep
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import pyzed.sl as sl

from p2_teleop.agents.quest_p2_arm_hfg_agent import QuestP2ArmHFGAgent
# from p2_teleop.constants import CAM_NAMES, LEFT_WRIST_CAM_ID, SCENE_CAM_ID
from p2_teleop.p2 import diana_api
from p2_teleop.p2.bg_p2_utils import blocking_moveJ, blocking_moveL
from p2_teleop.p2.p2_math_utils import (
    absolute_action2diana_pose,
    axangle2mat,
    axangles_rot_error,
    convert_pose_to_torso_frame,
)

from p2_teleop.p2.p2_math_utils import AngleType


class ZEDCameraWrapper:
    def __init__(self, serial_number: int, is_gmsl: bool = False, sender_ip: str = None, port: int = None, cam_name: str = "hfg_"):
        self.serial_number = serial_number
        self.is_gmsl = is_gmsl
        self.sender_ip = sender_ip
        self.port = port
        self.camera = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.latest_image = None
        self.cam_name = cam_name

    def open(self) -> bool:
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER

        if self.is_gmsl:
            init_params.set_from_stream(sender_ip=self.sender_ip, port=self.port)

        status = self.camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to open camera (SN={self.serial_number}): {status}")
            return False

        return True

    def start_grabbing(self):
        self.running = True
        self.thread = Thread(target=self._grab_loop)
        self.thread.daemon = True
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
                with self.lock:
                    self.latest_image = self.image.get_data().copy()
            sleep(0.0001)

    def get_latest_image(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_image.copy() if self.latest_image is not None else None

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.camera.close()


class ZEDMultiCameraSystem:
    def __init__(self, camera_configs: List[Dict]):
        self.cameras: Dict[int, ZEDCameraWrapper] = {
            config['serial']: ZEDCameraWrapper(
                serial_number=config['serial'],
                is_gmsl=config.get('is_gmsl', False),
                sender_ip=config.get('sender_ip'),
                port=config.get('port'),
                cam_name=config.get('cam_name')  # Default camera name prefix
            )
            for config in camera_configs
        }

    def setup(self):
        for sn, cam in self.cameras.items():
            if not cam.open():
                raise RuntimeError(f"Could not initialize ZED camera with serial: {sn}")
            cam.start_grabbing()

    def grab_all(self) -> Dict[int, Optional[np.ndarray]]:
        return {
            sn: cam.get_latest_image()
            for sn, cam in self.cameras.items()
        }

    def shutdown(self):
        for cam in self.cameras.values():
            cam.close()


class P2Env:
    def __init__(
        self,
        freq,
        is_delta_actions: bool,
        angle_type: AngleType,
        use_blocking_move: bool,
        sleep_after_move: float,
        left_reset_config,
        right_reset_config,
        yawing_gripper_reset_position: float,
        modality_config: Dict,
        outer_loop_delay=0.0005,
        servo_dt=0.2,
        ah_t=0.1,
        gain=20,
    ):
        ##################################

        # self.left_agent = QuestP2ArmHFGAgent(
        #     which_hand="r",
        #     which_robot="l",
        #     event_manager=None,
        #     oculus_reader=None,
        #     start=False,
        #     yawing_gripper_reset_position=yawing_gripper_reset_position,
        # )


        ##################################
        self.use_blocking_move = use_blocking_move
        self.sleep_after_move = sleep_after_move
        self.freq = freq
        self.angle_type = angle_type
        self.modality_config = modality_config
        self.left_reset_config = left_reset_config
        self.right_reset_config = right_reset_config
        self.is_delta_actions = is_delta_actions | False  # Default to False if not specified

        self.last_t = perf_counter()
        self.outer_loop_delay = outer_loop_delay

        self.camera_configs = [
            {'serial': 57366562, 'is_gmsl': True, 'sender_ip': '192.168.10.40', 'port': 40000, 'cam_name': 'hfg_left'},   # GMSL camera 0
            {'serial': 54832066, 'is_gmsl': True, 'sender_ip': '192.168.10.40', 'port': 30000, 'cam_name': 'hfg_right'},   # GMSL camera 1
        ]
        self.camera_system = ZEDMultiCameraSystem(self.camera_configs)
        self.camera_system.setup()

        self.obs_space_dict = {}
        self.action_space_dict = {}
        for key, val in self.modality_config.items():
            if key.startswith("video.") or key.startswith("state."):
                for modality in val.modality_keys:
                    self.obs_space_dict[modality] = None # Placeholder, will be set later
            elif key.startswith("action."):
                for modality in val.modality_keys:
                    self.action_space_dict[modality] = None

        self.servo_dt = servo_dt
        self.ah_t = ah_t
        self.gain = gain
        self.scale = 1.0

    # def start_recording(self, video_dir, video_name_no_ext):
    #     video_paths = {}
    #     for cap in self.caps.values():
    #         path = cap.start_recording(video_dir, video_name_no_ext)
    #         video_paths[cap.cam_id] = path
    #     return video_paths

    # def end_recording(self):
    #     for cap in self.caps.values():
    #         cap.end_recording()

    def start_valve(self):
        if self.left_agent.valve is not None:
            self.left_agent.valve.start()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        self.last_t = perf_counter()

        print("Resetting P2Env...")

        # self.left_agent._hfg_grip(grip=False)
        # self.left_agent.reset_yawing_motor_pose()
        # blocking_moveJ(
        #     self.left_reset_config,
        #     v=0.3,
        #     a=0.3,
        #     ip=self.left_agent.ip,
        #     timeout=15.0,
        # )

        # self.right_agent.get_hand_open_closed_qs()
        # self.right_agent.blocking_open_in_passthrough()
        # TODO: enable reset right arm joint positions once we have that in the state
        # blocking_moveJ(
        #     self.right_reset_config,
        #     v=0.1,
        #     a=0.1,
        #     ip=self.right_agent.ip,
        #     timeout=15.0,
        # )

        # By sleeping here we hopefully ensure the robot is stationary, meaning that the first image
        # we record is more consistent then if we take the first image while the robot is still moving from the reset.
        sleep(1.0)
        obs = self.get_obs()
        return obs, {}

    def get_obs(self):
        current_obs = self.obs_space_dict.copy()

        # left_agent_obs = self.left_agent.get_obs()

        images = self.camera_system.grab_all()
        left_arm_pose = self.get_current_pose()

        images = self.camera_system.grab_all()
        left_arm_pos, left_arm_rot = self.get_current_pose()
        # left_agent_obs = self.left_agent.get_obs()

        # Ensure images are in the correct shape and dtype
        if images[57366562] is not None:
            current_obs['video.hfg_left'] = images[57366562][None, :, :, :3].astype(np.uint8)
        else:
            current_obs['video.hfg_left'] = np.zeros((1, 1200, 1920, 3), dtype=np.uint8)

        if images[54832066] is not None:
            current_obs['video.hfg_right'] = images[54832066][None, :, :, :3].astype(np.uint8)
        else:
            current_obs['video.hfg_right'] = np.zeros((1, 1200, 1920, 3), dtype=np.uint8)

        # Ensure state vectors are (1, 3) arrays
        current_obs['state.ee_pos'] = np.array(left_arm_pos, dtype=np.float32).reshape(1, 3)
        current_obs['state.ee_rot'] = np.array(left_arm_rot, dtype=np.float32).reshape(1, 3)
        # current_obs['state.wrist'] = np.array([left_agent_obs['wrist_pressure_bar'], left_agent_obs['wrist_gripper_proximity'], left_agent_obs['wrist_cup_sensor']], dtype=np.float32).reshape(1, 3)
        current_obs['state.wrist'] = np.array([np.random.rand(), np.random.rand(), np.random.rand()], dtype=np.float32).reshape(1, 3)

        # right_agent_obs = self.right_agent.get_obs()

        return current_obs

    def step(self, action) -> Tuple[Dict, float, bool, bool, dict]:
        # In training, Octo uses +1 to mean gripper is open, which is how valve position works already.
        action_valve = action['action.valve'][0]
        action_ee_pos = action['action.ee_pos'][0]
        action_ee_rot = action['action.ee_rot'][0]

        valve_position = np.clip(action_valve, 0.0, 1.0)

        # if self.left_agent.valve is not None:
        #     self.left_agent.valve.command_position(
        #         position=valve_position, slew_rate_scale=1.0
        #     )

        if self.is_delta_actions:
            # TODO: re-observing the current pose might be a bad idea, since the obs the policy used
            #  to predict the actions was the one returned by the previous call to step().
            start_pos, start_rot = self.get_current_pose()
            ee_diana_pose, ee_pos, ee_rot = delta_action_vec2diana_pose(
                action, start_pos, start_rot, self.angle_type
            )
        else:
            action_vec = np.concatenate([action['action.ee_pos'][0], action['action.ee_rot'][0]])
            ee_diana_pose, ee_pos, ee_rot = absolute_action2diana_pose(
                action_vec[0:6], self.angle_type
            )

        # self.viz_current_pose()

        # current_observed_pose_debug = np.zeros(6)
        # diana_api.getTcpPos(current_observed_pose_debug, self.left_agent.ip)
        # delta_axangle = axangles_rot_error(
        #     ee_diana_pose[3:], current_observed_pose_debug[3:]
        # )
        # rr.log("delta_axangle", rr.Scalar(delta_axangle))
        # if delta_axangle > np.deg2rad(15):
        #     print("Large change in angle!!!")
        #     obs = self.get_obs()
        #     if self.left_agent.valve is not None:
        #         self.left_agent.valve.command_position(
        #             position=1.0, slew_rate_scale=1.0
        #         )
        #     return obs, 0.0, True, False, {}

        if self.use_blocking_move:
            # blocking_moveL(
            #     ee_diana_pose,
            #     a=0.3,
            #     v=0.3,
            #     ip=self.left_agent.ip,
            #     timeout=15,
            # )
            # if self.sleep_after_move > 0:
            #     sleep(self.sleep_after_move)
            print("Blocking moveL")
        else:
            diana_api.servoL(
                ee_diana_pose,
                t=self.servo_dt,
                ah_t=self.ah_t,
                gain=self.gain,
                scale=self.scale,
                ipAddress=self.left_agent.ip,
            )

        # Wait for the robot to move. This is mostly for the servoL non-blocking call.
        period = 1 / self.freq
        sleep_dt = period - (perf_counter() - self.last_t) - self.outer_loop_delay
        # rr.log("sleep_dt", rr.Scalar(sleep_dt))
        if sleep_dt > 0:  # makes the Y axis scaling less jumpy
            sleep(sleep_dt)

        self.last_t = perf_counter()

        obs = self.get_obs()
        is_done = False

        T_torso2ee = convert_pose_to_torso_frame(obs["state.ee_pos"], obs["state.ee_rot"])
        if T_torso2ee[1, 3] < 0.03 and valve_position > 0.8:
            print(
                "Flagging policy as done due to having reached "
                "place side and released item."
            )
            is_done = True

        return obs, 0.0, is_done, False, {}

    def viz_current_pose(self):
        pass

    def get_current_pose(self):
        current_observed_pose = np.zeros(6)
        # diana_api.getTcpPos(current_observed_pose, self.left_agent.ip)
        current_observed_pose = np.random.rand(6)  # Simulating a random pose for testing

        # current_pos = current_observed_pose[0:3]
        # current_rot = axangle2mat(current_observed_pose[3:6])
        # return current_pos, current_rot
        return current_observed_pose[0:3], current_observed_pose[3:6]

    def close(self):
        for cap in self.caps.values():
            cap.stop()
            cap.join()
        self.left_agent.close()
        self.right_agent.close()
        self.waist_agent.close()