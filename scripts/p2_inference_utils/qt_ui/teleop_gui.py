from enum import Enum

from PyQt5 import uic
from PyQt5.QtCore import QSettings, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget, QLabel, QFormLayout, QStyle

from p2_teleop.benchmarking.db_utils import get_gripper_db_collection
from p2_teleop.constants import UI_ROOT, CAMERA_CONFIGS
from p2_teleop.p2_teleop_app import Teleop
from p2_teleop.qt_ui.camera_object import CameraObject
from p2_teleop.qt_ui.dialogs import (
    EndToteDialog,
    StartToteDialog,
    StartEpisodeDialog,
    EndEpisodeDialog,
)
from p2_teleop.qt_ui.hfg_gui import HighFlowGripperGui
from p2_teleop.qt_ui.qtoast_notification import QToastNotification
from p2_teleop.qt_ui.tote_divisions_widget import ToteDivisionsWidget


class GraspPrimitiveWidget(QWidget):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.form = QFormLayout()
        self.setLayout(self.form)
        for i, name in enumerate(options):
            option_label = QLabel(parent=self, text=name)
            option_label.setEnabled(False)

            icon_label = QLabel(parent=self)
            icon_label.setEnabled(False)
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            icon_label.setPixmap(icon.pixmap(12, 12))

            icon_label.setVisible(i == 0)
            self.form.addRow(icon_label, option_label)

    def set_option(self, idx: int):
        for i in range(self.form.rowCount()):
            icon_label = self.form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget()
            icon_label.setVisible(i == idx)


class StartStopCancelButton(QObject):
    """Helper class to handle the state of the start/stop buttons, where 'start' launches a cancelable dialog."""

    on_start = pyqtSignal()
    on_stop = pyqtSignal()

    def __init__(
        self,
        button,
        start_slot,
        start_accepted_signal,
        end_slot,
        end_accepted_signal,
        suffix="",
    ):
        super().__init__(parent=button)
        self.button = button
        self.suffix = suffix

        self.is_start = True
        self.button.setStyleSheet(
            "border: 2px solid green; border-radius: 3px; padding: 1px;"
        )

        self.button.clicked.connect(self.toggle)

        # When the button is clicked in the start state, we emit the on_start signal which calls the start_slot.
        # Then a dialog will appear and if the user clicks 'OK', the start_accepted_signal will be emitted,
        # which will call the on_start_accepted slot and change the state of the button.
        # Otherwise, we do not change the state of the button.
        # Then, when the button is clicked in the stop state, we emit the on_stop signal which calls the end_slot.
        # Then a dialog will appear and if the user clicks 'OK', the end_accepted_signal will be emitted,
        # which will call the on_stop_accepted slot and change the state of the button.
        self.on_start.connect(start_slot)
        start_accepted_signal.connect(self.on_start_accepted)
        self.on_stop.connect(end_slot)
        end_accepted_signal.connect(self.on_stop_accepted)

    def toggle(self):
        if self.is_start:
            self.on_start.emit()
        else:
            self.on_stop.emit()

    def on_start_accepted(self):
        self.is_start = False
        self.button.setText(f"Stop {self.suffix}")
        self.button.setStyleSheet(
            "border: 2px solid red; border-radius: 3px; padding: 1px;"
        )

    def on_stop_accepted(self):
        self.is_start = True
        self.button.setText(f"Start {self.suffix}")
        self.button.setStyleSheet(
            "border: 2px solid green; border-radius: 3px; padding: 1px;"
        )


def get_expected_cam_for_gripper_id(gripper_id: str):
    grippers_df = get_gripper_db_collection()
    return grippers_df[gripper_id == grippers_df["gripper_id"]][
        "expected_wrist_cam"
    ].item()


def form_camera_checkbox_name(cam_name):
    return f"{cam_name}_checkbox"


class TeleopMainWindow(QWidget):
    episode_started = pyqtSignal(dict)
    episode_ended = pyqtSignal(dict)
    tote_started = pyqtSignal(dict)
    tote_ended = pyqtSignal(dict)

    def __init__(self, teleop: Teleop):
        super().__init__()
        self.teleop = teleop
        self.ui = uic.loadUi(UI_ROOT / "main_window.ui", self)

        # Create camera objects and store in dict by cam_name
        self.cameras = {}
        for config in CAMERA_CONFIGS:
            checkbox = getattr(self.ui, form_camera_checkbox_name(config["cam_name"]))
            cam_obj = CameraObject(
                node=self.teleop.node,
                cam_name=config["cam_name"],
                start_service=config["start_service"],
                stop_service=config["stop_service"],
                camera_type=config["camera_type"],
                checkbox=checkbox,
            )
            self.cameras[config["cam_name"]] = cam_obj
            checkbox.toggled.connect(
                lambda checked, cam=config["cam_name"]: self.on_checkbox_toggled(
                    cam, checked
                )
            )

        if (
            self.teleop.agents.right_agent is not None
            and self.teleop.agents.right_agent.grasp_cycler is not None
        ):
            grasp_primitive_names = [
                n for (n, _, _) in self.teleop.agents.right_agent.grasp_cycler.options
            ]
        else:
            grasp_primitive_names = []
        self.hand_grasp_primitive_widget = GraspPrimitiveWidget(
            self.ui.right_group, grasp_primitive_names
        )
        self.ui.right_group.layout().insertWidget(2, self.hand_grasp_primitive_widget)

        # Create the Tote Divisions widget to indicate which tote division we're choosing from.
        # this connects signals/slots so that the Tote Division widget UI gets initialized based on the initial
        # Tote ID from the Tote ID Combobox. It also sets the tab order so that the Tote Division widget is
        # selected after the Tote ID Combobox, even when a new Tote ID is selected.
        self.tote_division_widget = ToteDivisionsWidget(
            self.ui.recording_group, self.ui.tote_divisions_layout
        )
        self.tote_division_widget.setEnabled(False)

        self.tote_stop_start_button = StartStopCancelButton(
            button=self.ui.start_stop_tote_button,
            start_slot=self.open_start_tote_dialog,
            start_accepted_signal=self.tote_started,
            end_slot=self.open_end_tote_dialog,
            end_accepted_signal=self.tote_ended,
            suffix="Tote",
        )

        self.episode_stop_start_button = StartStopCancelButton(
            button=self.ui.start_stop_episode_button,
            start_slot=self.open_start_episode_dialog,
            start_accepted_signal=self.episode_started,
            end_slot=self.open_end_episode_dialog,
            end_accepted_signal=self.episode_ended,
            suffix="Episode",
        )

        self.hfg_gui = HighFlowGripperGui(
            self, self.ui, self.teleop.agents.left_agent.valve
        )
        self.ui.home_yawing_motor_button.clicked.connect(
            self.teleop.agents.left_agent.home_yawing_gripper
        )
        self.ui.reset_yawing_motor_button.clicked.connect(
            self.teleop.agents.left_agent.reset_yawing_motor_pose
        )
        # TODO: show whether or not the valve is connected! or maybe a button that tests it?
        self.restore_settings()

        # This is a member variable to keep the toast notification alive
        self.toast = None

        self.save_settings_timer = QTimer(self)
        self.save_settings_timer.timeout.connect(self.save_settings)
        self.save_settings_timer.start(30_000)

    def on_checkbox_toggled(self, cam_name: str, checked: bool):
        camera = self.cameras[cam_name]
        if checked:
            start_ready = camera.start_client.service_is_ready()
            stop_ready = camera.stop_client.service_is_ready()
            if not (start_ready and stop_ready):
                camera.checkbox.setChecked(False)
                camera.checkbox.setToolTip("ZED services not available")
                self.show_notification(
                    f"{cam_name.capitalize()} camera service not available"
                )
            else:
                camera.checkbox.setToolTip("")

    def open_start_episode_dialog(self):
        dialog = StartEpisodeDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            if data is None:
                self.ui.recording_log_edit.insertPlainText("Unknown product!\n")
                return

            for camera_name, camera_obj in self.cameras.items():
                if not camera_obj.checkbox.isChecked():
                    camera_obj.checkbox.setChecked(True)
                    self.on_checkbox_toggled(camera_name, True)

            self.episode_started.emit(data)
            self.update_metadata(data)
            self.ui.next_repeat_button.setEnabled(True)
            # You shouldn't be able to end a tote before stopping an episode.
            self.ui.start_stop_tote_button.setEnabled(False)
            self.ui.start_stop_episode_button.setToolTip(
                "Before ending the tote, end the episode"
            )

    def update_metadata(self, data):
        self.ui.metadata_group.setEnabled(True)
        self.ui.tote_type_edit.setText(data["tote_type"])
        self.tote_division_widget.update_divisions(data["tote_type"])
        self.tote_division_widget.set_selection(data["division_id"])
        self.ui.product_id_edit.setText(data["product_id"])
        self.ui.pack_type_edit.setText(data["pack_type"])
        self.ui.gripper_id_edit.setText(data["gripper_id"])
        self.ui.operator_id_edit.setText(data["operator_id"])
        self.ui.task_type_edit.setText(data["task_type"])

    def open_end_episode_dialog(self):
        dialog = EndEpisodeDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            self.episode_ended.emit(data)
            self.ui.next_repeat_button.setEnabled(False)
            self.ui.start_stop_tote_button.setEnabled(True)
            self.ui.start_stop_episode_button.setToolTip("")

    def open_start_tote_dialog(self):
        dialog = StartToteDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            self.tote_started.emit(data)
            # automatically start an episode
            self.open_start_episode_dialog()

    def open_end_tote_dialog(self):
        dialog = EndToteDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            self.tote_ended.emit(data)

    def on_data_recording_log(self, msg: str):
        self.ui.recording_log_edit.insertPlainText(msg + "\n")

    def show_notification(self, msg: str):
        self.toast = QToastNotification(self, msg)
        self.toast.show()

    def restore_settings(self):
        # Force all checkboxes off initially
        for config in CAMERA_CONFIGS:
            getattr(self.ui, form_camera_checkbox_name(config["cam_name"])).setChecked(
                False
            )
        settings = self.get_settings()
        if hasattr(self.ui, "hsplit") and settings.contains("hsplitter_state"):
            self.ui.hsplit.restoreState(settings.value("hsplitter_state"))

    def save_settings(self):
        settings = self.get_settings()
        for config in CAMERA_CONFIGS:
            settings.setValue(
                form_camera_checkbox_name(config["cam_name"]),
                getattr(
                    self.ui, form_camera_checkbox_name(config["cam_name"])
                ).isChecked(),
            )
        if hasattr(self.ui, "hsplit"):
            settings.setValue("hsplitter_state", self.ui.hsplit.saveState())

    def get_settings(self):
        return QSettings("BG", "P2Teleop")

    def closeEvent(self, event):
        self.save_settings()
        for agent in self.teleop.agents.as_list():
            agent.close()
        for cam in self.cameras.values():
            cam.shutdown()
        event.accept()

    def set_recording_size(self, num_state_action_steps: int):
        self.ui.recording_size_spinbox.setValue(num_state_action_steps)
