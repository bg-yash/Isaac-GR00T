from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QCheckBox
import rclpy
from rclpy.node import Node
from zed_msgs.srv import StartSvoRec
from std_srvs.srv import Trigger


class CameraObject(QObject):
    service_available_signal = pyqtSignal(bool)

    def __init__(
        self,
        node: Node,
        cam_name: str,
        start_service: str,
        stop_service: str,
        camera_type: str,
        checkbox: QCheckBox = None,
        check_interval_ms: int = 2000,
    ):
        super().__init__()
        self.node = node
        self.cam_name = cam_name
        self.camera_type = camera_type
        self.checkbox = checkbox

        self.start_client = self.node.create_client(StartSvoRec, start_service)
        self.stop_client = self.node.create_client(Trigger, stop_service)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_services)
        self.timer.start(check_interval_ms)

    def check_services(self):
        if self.node is None or not rclpy.ok():
            return

        try:
            start_ready = self.start_client.service_is_ready()
            stop_ready = self.stop_client.service_is_ready()
            available = start_ready and stop_ready
        except Exception as e:
            if self.node:
                self.node.get_logger().error(
                    f"[{self.cam_name}] Error checking service: {e}"
                )
            return

        self.service_available_signal.emit(available)

        if self.checkbox:
            try:
                self.checkbox.setToolTip(
                    "" if available else "ZED services not available"
                )
            except RuntimeError:
                self.checkbox = None

    def start_recording(self, svo_filename: str):
        if not self.start_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(
                f"[{self.cam_name}] Start service not available!"
            )
            return

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
        else:
            self.node.get_logger().error(
                f"[{self.cam_name}] Failed to start SVO recording."
            )

    def stop_recording(self):
        if not self.stop_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error(
                f"[{self.cam_name}] Stop service not available!"
            )
            return

        req = Trigger.Request()
        future = self.stop_client.call_async(req)

        while not future.done() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        result = future.result()
        if result and result.success:
            self.node.get_logger().info(f"[{self.cam_name}] SVO recording stopped.")
        else:
            self.node.get_logger().error(
                f"[{self.cam_name}] Failed to stop SVO recording."
            )

    def shutdown(self):
        self.timer.stop()
        try:
            if self.checkbox:
                self.checkbox.toggled.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.checkbox = None
        self.node = None
