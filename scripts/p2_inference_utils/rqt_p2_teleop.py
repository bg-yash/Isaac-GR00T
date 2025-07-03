import os
import signal
from argparse import Namespace
from pathlib import Path
from threading import Thread
from functools import partial

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication
from rqt_gui_py.plugin import Plugin
import rclpy
from p2_teleop.constants import (
    DEFAULT_LEFT_IP,
    DEFAULT_RIGHT_IP,
    DEFAULT_VALVE_IP,
    DEFAULT_YAWING_GRIPPER_RESET_POSITION,
    LEFT_START_CONFIG,
    RIGHT_START_CONFIG,
)
from p2_teleop.p2_teleop_app import ArmChoice, ControllerChoice, Teleop
from p2_teleop.qt_ui.teleop_gui import TeleopMainWindow
from p2_teleop.qt_ui.dialogs import StartEpisodeDialog
from p2_teleop.episode_recording.episode_recorder import RecorderManager
from p2_teleop.benchmarking.benchmarking_manager import BenchmarkingManager
from p2_teleop.p2.bg_p2_utils import blocking_moveJ
from p2_teleop.teleop_user_controls import SingleClick, ButtonType


def spin_node_forever(node):
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass


class TeleopSignals(QObject):
    set_left_control_frame = pyqtSignal(str)
    set_right_control_frame = pyqtSignal(str)
    set_hand_grasp_primitive = pyqtSignal(int)
    show_notification = pyqtSignal(str)


class TeleopWorker:
    def __init__(
        self, teleop: Teleop, recorder_manager: RecorderManager, parent: QObject
    ):
        self.teleop = teleop
        self.recorder_manager = recorder_manager
        self.parent = parent
        self.signals = TeleopSignals(parent)

        for agent in self.teleop.agents.as_list():
            agent.reset_event.append_callback(
                lambda _: self.show_reset_notification(agent)
            )

        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.moveToThread(QApplication.instance().thread())
        self.timer.start(0)

    def show_reset_notification(self, agent):
        self.signals.show_notification.emit(f"Resetting {agent.which_robot} arm")

    def step(self):
        try:
            observations, actions = self.teleop.step()
            self.recorder_manager.step(observations, actions)
            # print("Step executed")
        except Exception as e:
            print(f"Error in step: {e}")

        if left := self.teleop.agents.left_agent:
            self.signals.set_left_control_frame.emit(
                left.control_frame_switcher.get().name
            )
        if right := self.teleop.agents.right_agent:
            self.signals.set_right_control_frame.emit(
                right.control_frame_switcher.get().name
            )
            self.signals.set_hand_grasp_primitive.emit(right.grasp_cycler.idx)


class QBlockingMoveThread(QThread):
    def __init__(self, parent, ip: str, q, v=0.3, a=0.3, timeout=15):
        super().__init__(parent)
        self.ip = ip
        self.q = q
        self.v = v
        self.a = a
        self.timeout = timeout

    def run(self):
        blocking_moveJ(self.q, self.v, self.a, ip=self.ip, timeout=self.timeout)


class QRobotReset(QObject):
    def __init__(self, agents, parent: QObject):
        super().__init__(parent)
        self.agents = agents

    def reset_left_side(self):
        self.agents.left_agent._hfg_grip(grip=False)
        self.agents.left_agent.reset_yawing_motor_pose()
        move = QBlockingMoveThread(self, self.agents.left_agent.ip, LEFT_START_CONFIG)
        move.start()

    def reset_right_side(self):
        self.agents.right_agent.get_hand_open_closed_qs()
        self.agents.right_agent.blocking_open_in_passthrough()
        move = QBlockingMoveThread(self, self.agents.right_agent.ip, RIGHT_START_CONFIG)
        move.start()


def managed_repeat_start_next_episode(
    teleop_gui: TeleopMainWindow,
    recorder: RecorderManager,
    benchmarker: BenchmarkingManager,
):
    was_recording = recorder.recording_enabled
    if was_recording:
        end_episode_data = {"transfer_termination": "SUCCESS"}
        recorder.end_episode(end_episode_data)

    # TODO(2025/05/30, Dylan Colli): Do we want to remove benchmarking totally?
    if benchmarker.is_benchmarking:
        raise NotImplementedError(
            "Need to add support for episode repeat when benchmarking"
        )

    if was_recording:
        # TODO(2025/05/30, Yash Shukla): Potentially put this in EpisodeRecorder?
        start_dialog = StartEpisodeDialog(teleop_gui)
        start_dialog.restore_settings()
        start_data = start_dialog.get_data()

        teleop_gui.episode_started.emit(start_data)
        teleop_gui.update_metadata(start_data)


class RqtP2Teleop(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self.setObjectName("RqtP2Teleop")

        self.node = rclpy.create_node("p2_teleop_rqt_node")
        self.spin_thread = Thread(
            target=spin_node_forever, args=(self.node,), daemon=True
        )
        self.spin_thread.start()

        signal.signal(signal.SIGINT, self._shutdown_from_signal)

        args = Namespace(
            data_root=Path("/opt/bg/ws/src/bg_p2/p2_teleop/data"),
            no_gui=False,  # Run without GUI
            arm=ArmChoice.LEFT,  # Whether to control both arms, left, or right arm
            controller=ControllerChoice.R,  # Which Quest3 controller to use for controlling the arm.Only applies when using a single arm with --arm option
            left_ip=DEFAULT_LEFT_IP,  # Left arm controller IP address
            right_ip=DEFAULT_RIGHT_IP,  # Right arm controller IP address
            valve_ip=DEFAULT_VALVE_IP,  # PCM valve IP address
            min_grasp_valve_position=0.5,  # the minimum position of the HFG valve to use when barely pressing the quest trigger. 1.0 corresponds to 0%% air flow, i.e. no suction
            min_grasp_valve_position_limit=1.0,  # the minimum position of the HFG valve to use when *not* pressing the quest trigger. Must be >= min-grasp-valve-position. 1.0 corresponds to 0%% air flow, i.e. no suction.
            max_grasp_valve_position=0.1,  # the maximum position of the HFG valve to use when almost fully pressing the quest trigger. 0.0 corresponds to 100%% air flow, i.e. full suction
            max_grasp_valve_position_limit=0.0,  # the maximum valve position of the HFG valve to use when *fully* pressing the quest trigger. Must be <= max-grasp-valve-position. 0.0 corresponds to 100%% air flow, i.e. full suction
            rerun=True,  # Enable rerun logging
            dry_run=False,  # Dry run mode
            yawing_gripper_reset_position=DEFAULT_YAWING_GRIPPER_RESET_POSITION,
            pos_scale=None,  # Scale up the delta position commands to the robot. 1x is no scaling. Larger numbers make the robot move faster/more given the same movement of the controller. Smaller numbers make it move slower/less.",
            no_waist=True,  # Disable the waist agent, since for the p2.1L the waist is not currently working
        )

        self.teleop_app = Teleop(args, node=self.node)
        self.main_widget = TeleopMainWindow(self.teleop_app)

        self.recorder_manager = RecorderManager(
            self.main_widget, self.teleop_app, args.data_root
        )
        self.benchmarking_manager = BenchmarkingManager(
            self.main_widget, root=args.data_root
        )

        if not args.dry_run:
            self.worker = TeleopWorker(
                self.teleop_app, self.recorder_manager, parent=self.main_widget
            )
            self.worker.signals.set_left_control_frame.connect(
                self.main_widget.ui.left_control_frame_edit.setText
            )
            self.worker.signals.set_right_control_frame.connect(
                self.main_widget.ui.right_control_frame_edit.setText
            )
            self.worker.signals.set_hand_grasp_primitive.connect(
                self.main_widget.hand_grasp_primitive_widget.set_option
            )
            self.worker.signals.show_notification.connect(
                self.main_widget.show_notification
            )
        else:
            self.worker = None

        self.recorder_manager.data_recording_log.connect(
            self.main_widget.on_data_recording_log
        )
        self.main_widget.episode_started.connect(self.recorder_manager.start_episode)
        self.main_widget.episode_ended.connect(self.recorder_manager.end_episode)

        self.recorder_manager.episode_started.connect(
            self.benchmarking_manager.on_episode_start
        )
        self.main_widget.episode_ended.connect(self.benchmarking_manager.on_episode_end)
        self.main_widget.tote_started.connect(self.benchmarking_manager.on_tote_start)
        self.main_widget.tote_ended.connect(self.benchmarking_manager.on_tote_end)

        self.main_widget.ui.next_repeat_button.clicked.connect(
            partial(
                managed_repeat_start_next_episode,
                self.main_widget,
                self.recorder_manager,
                self.benchmarking_manager,
            )
        )

        self._set_up_robot_reset_controls()

        if context.serial_number() > 1:
            self.main_widget.setWindowTitle(
                f"{self.main_widget.windowTitle()} ({context.serial_number()})"
            )

        context.add_widget(self.main_widget)

    def _set_up_robot_reset_controls(self):
        reset_helper = QRobotReset(self.teleop_app.agents, parent=self.main_widget)

        self.main_widget.ui.move_left_to_start_button.clicked.connect(
            reset_helper.reset_left_side
        )
        self.main_widget.ui.move_right_to_start_button.clicked.connect(
            reset_helper.reset_right_side
        )

        def _reset_left(button_data):
            if button_data["rightTrig"][0] == 0.0:
                reset_helper.reset_left_side()
                managed_repeat_start_next_episode(
                    self.main_widget, self.recorder_manager, self.benchmarking_manager
                )

        def _reset_right(button_data):
            if button_data["leftTrig"][0] == 0.0:
                reset_helper.reset_right_side()
                managed_repeat_start_next_episode(
                    self.main_widget, self.recorder_manager, self.benchmarking_manager
                )

        self.teleop_app.event_manager.add_event(
            SingleClick(ButtonType.JOYSTICK, "l", _reset_right, 0, timeout=0.4)
        )
        self.teleop_app.event_manager.add_event(
            SingleClick(ButtonType.JOYSTICK, "r", _reset_left, 0, timeout=0.4)
        )

    def _shutdown_from_signal(self, *_):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.node:
            self.node.get_logger().info("Caught SIGINT, shutting down...")

        self.shutdown_plugin()

        if rclpy.ok():
            rclpy.shutdown()

        QApplication.quit()

    def shutdown_plugin(self):
        self.main_widget.save_settings()
        for agent in self.teleop_app.agents.as_list():
            agent.close()
        self.node.destroy_node()

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass
