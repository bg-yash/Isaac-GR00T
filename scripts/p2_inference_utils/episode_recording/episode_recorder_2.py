import rclpy
import time
from pathlib import Path
from datetime import datetime
from rclpy.node import Node
from zed_msgs.srv import StartSvoRec
from std_srvs.srv import Trigger

class CameraObject:
    def __init__(
        self,
        node: Node,
        cam_name: str,
        start_service: str,
        stop_service: str,
        camera_type: str = "zed"
    ):
        self.node = node
        self.cam_name = cam_name
        self.camera_type = camera_type

        self.start_client = self.node.create_client(StartSvoRec, start_service)
        self.stop_client = self.node.create_client(Trigger, stop_service)

    def start_recording(self, svo_filename: str):

        req = StartSvoRec.Request()
        req.svo_filename = svo_filename
        req.compression_mode = 0  # H.265
        req.input_transcode = False
        req.target_framerate = 60

        future = self.start_client.call_async(req)

        while not future.done() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        result = future.result()
        if result and result.success:
            self.node.get_logger().info(f"[{self.cam_name}] SVO recording started.")
            return True
        else:
            self.node.get_logger().error(
                f"[{self.cam_name}] Failed to start SVO recording."
            )
            return False

    def stop_recording(self):

        req = Trigger.Request()
        future = self.stop_client.call_async(req)

        while not future.done() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        result = future.result()
        if result and result.success:
            self.node.get_logger().info(f"[{self.cam_name}] SVO recording stopped.")
            return True
        else:
            self.node.get_logger().error(
                f"[{self.cam_name}] Failed to stop SVO recording."
            )
            return False


class EpisodeRecorder:
    def __init__(self, node, camera_config):
        self.node = node
        self.camera_objects = {
            name: CameraObject(
                node=node,
                cam_name=name,
                start_service=cfg["start"],
                stop_service=cfg["stop"]
            )
            for name, cfg in camera_config.items()
        }

    def start_episode(self, episode_dir: Path):
        episode_dir.mkdir(parents=True, exist_ok=True)
        self.node.get_logger().info(f"Starting episode: {episode_dir}")

        for name, cam in self.camera_objects.items():
            svo_path = episode_dir / f"{name}.svo2"
            success = cam.start_recording(str(svo_path))
            if not success:
                self.node.get_logger().warning(f"Skipping {name} due to failure.")

    def end_episode(self):
        for name, cam in self.camera_objects.items():
            success = cam.stop_recording()
            if not success:
                self.node.get_logger().warning(f"Could not stop {name} properly.")


def main():
    rclpy.init()
    node = rclpy.create_node("episode_recorder")

    camera_config = {
        "cam1": {
            "start": "/sensors/cam1/start_svo_rec",
            "stop": "/sensors/cam1/stop_svo_rec"
        },
        # "cam2": {
        #     "start": "/sensors/cam2/start_svo_rec",
        #     "stop": "/sensors/cam2/stop_svo_rec"
        # },
    }

    recorder = EpisodeRecorder(node, camera_config)


    for i in range(4):
        date_str = datetime.now().strftime("%H-%M-%S")
        episode_dir = Path("/opt/bg/ws/teleop_episodes").expanduser() / date_str
        recorder.start_episode(episode_dir)
        time.sleep(4)
        recorder.end_episode()

    node.get_logger().info("Episode complete.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
