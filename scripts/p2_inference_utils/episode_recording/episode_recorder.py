import json
import pickle
from datetime import datetime
from pathlib import Path
from time import time, perf_counter
from typing import Dict, List
from uuid import uuid4
from p2_teleop.constants import EPISODE_TIMEOUT_SECONDS

import git
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

from p2_teleop.agents.quest_arm_agent import ControlState
from p2_teleop.episode_recording.episode_utils import METADATA_JSON, DATA_PKL
from p2_teleop.filesystem_utils import mkdir_as_logs_group
from p2_teleop.p2 import diana_api
from p2_teleop.p2_teleop_app import Teleop
from p2_teleop.qt_ui.teleop_gui import TeleopMainWindow


class DataRecorder(QObject):
    def __init__(self, root: Path, parent: QObject):
        super().__init__(parent)
        self.root = root
        self.is_recording = False
        self.episode_data = []

    def on_start_episode(self, episode_dir: Path):
        """
        :param episode_dir: Path of the new episode. It will exist by this point.
        :return:
        """
        self.episode_data = []

    def on_end_episode(self, episode_dir: Path):
        """
        :param episode_dir: Path where the episode was saved.
        :return:
        """
        pass

    def on_teleop_step(self, observations: List[Dict], actions: List[Dict]):
        pass


class StateActionDataRecorder(DataRecorder):
    def on_teleop_step(
        self, observations: List[Dict], actions: List[Dict], active_status: bool
    ):
        step_data = {
            "time": time(),
            "obs": observations,
            "actions": actions,
            "is_active": active_status,
        }
        self.episode_data.append(step_data)

    def on_end_episode(self, episode_dir: Path):
        super().on_end_episode(episode_dir)
        with (episode_dir / DATA_PKL).open("wb") as f:
            pickle.dump(self.episode_data, f)


class RecorderManager(QObject):
    data_recording_log = pyqtSignal(str)
    episode_started = pyqtSignal(dict, Path)

    def __init__(self, teleop_gui: TeleopMainWindow, teleop_app: Teleop, root: Path):
        super().__init__(parent=teleop_gui)
        self.session_uuid = str(uuid4())
        self.teleop_gui = teleop_gui
        self.teleop_app = teleop_app

        self.left_agent = teleop_app.agents.left_agent
        self.right_agent = teleop_app.agents.right_agent

        self.root = (
            Path(root).expanduser()
            / "teleop_episodes"
            / datetime.now().strftime("%Y-%m-%d")
        )
        self.state_action_data_recorder = StateActionDataRecorder(
            self.root, parent=self
        )
        self.recorders = [self.state_action_data_recorder]

        self.recording_enabled = False
        self.was_active = False
        self.episode_idx = 0
        self.current_episode_root = None
        self.current_episode_metadata = None
        self.episode_start_time = None

    def start_episode(self, start_episode_data: Dict):
        """
        Add a unique uuid to the episode, tell the step() function it should start recording, and create a new folder
        for the episode.
        :param start_episode_data: Dict of data from the StartEpisodeDialog.
        :return:
        """
        self.episode_start_time = perf_counter()
        self.recording_enabled = True
        self.current_episode_root = self.get_next_episode_dir(start_episode_data)
        mkdir_as_logs_group(self.current_episode_root)

        self.current_episode_metadata = start_episode_data
        self.current_episode_metadata["uuid"] = str(uuid4())
        repo = git.Repo(search_parent_directories=True)
        self.current_episode_metadata["git"] = repo.head.object.hexsha
        self.current_episode_metadata["date"] = datetime.now().isoformat()

        for agent in self.teleop_app.agents.as_list():
            tcp_offset = np.zeros(6)
            diana_api.getDefaultActiveTcpPose(tcp_offset, ipAddress=agent.ip)
            self.current_episode_metadata[agent.which_robot] = (
                agent.get_episode_metadata()
            )

        self.save_episode_metadata()

        for recorder in self.recorders:
            recorder.on_start_episode(self.current_episode_root)

        self._start_svo_recordings()
        self.log(f"Starting episode {self.current_episode_root}")
        self.episode_started.emit(
            self.current_episode_metadata, self.current_episode_root
        )

    def end_episode(self, end_episode_data):
        self.recording_enabled = False
        self._stop_svo_recordings()

        self.current_episode_metadata.update(end_episode_data)
        self.save_episode_metadata()

        for recorder in self.recorders:
            recorder.on_end_episode(self.current_episode_root)

        self.log(f"Saved episode {self.current_episode_root}")

    def step(self, observations, actions):
        either_active = self.is_either_active()
        if self.recording_enabled:
            for recorder in self.recorders:
                recorder.is_recording = either_active and self.was_active
                recorder.on_teleop_step(observations, actions, either_active)
            self.teleop_gui.set_recording_size(
                len(self.state_action_data_recorder.episode_data)
            )

            time_since_episode_start = perf_counter() - self.episode_start_time
            if time_since_episode_start > EPISODE_TIMEOUT_SECONDS:
                self.log(
                    f"Episode running for more than {EPISODE_TIMEOUT_SECONDS} seconds, ending it."
                )
                end_data = {
                    "transfer_termination": "EPISODE_TIMEOUT",
                }
                self.teleop_gui.episode_ended.emit(end_data)

        else:
            for recorder in self.recorders:
                recorder.is_recording = False

        self.was_active = either_active

    def is_either_active(self):
        return any(
            (
                (
                    self.left_agent
                    and self.left_agent.control_state == ControlState.ACTIVE
                ),
                (
                    self.right_agent
                    and self.right_agent.control_state == ControlState.ACTIVE
                ),
            )
        )

    def get_next_episode_dir(self, start_episode_data):
        episode_identifier = "_".join(start_episode_data.values())
        date = datetime.now().strftime("%Y-%m-%d")
        episode_dir = (
            self.root
            / f"{date}_{episode_identifier}_episode_{self.episode_idx}_session_{self.session_uuid}"
        )
        self.episode_idx += 1
        return episode_dir

    def save_episode_metadata(self):
        with (self.current_episode_root / METADATA_JSON).open("w") as f:
            json.dump(self.current_episode_metadata, f, indent=2)

    def _handle_camera_recording(self, key, checkbox, start):
        """
        Start or stop SVO recording for a specific camera based on the checkbox state.
        """
        client = (
            self.teleop_gui.cameras[key].start_client
            if start
            else self.teleop_gui.cameras[key].stop_client
        )
        action = (
            self.teleop_gui.cameras[key].start_recording
            if start
            else self.teleop_gui.cameras[key].stop_recording
        )
        if checkbox.isChecked():
            if client.service_is_ready():
                if start:
                    action(str(self.current_episode_root / f"{key}.svo2"))
                else:
                    action()
            else:
                self.log(
                    f"{key.capitalize()} camera service not available. Skipping SVO {'start' if start else 'stop'}."
                )

    def _start_svo_recordings(self):
        """
        Starts SVO recording for all selected cameras.

        Note: (Yash Shukla 06/04/2025):
            This method performs sequential service calls to each camera's start_recording service.
            As the number of cameras grows, this can introduce a noticeable delay between the first
            and last camera's recording start times. Consider threading these calls in the future to
            minimize per-episode startup latency and improve synchronization across cameras.
        """
        self._handle_camera_recording(
            "base", self.teleop_gui.ui.base_checkbox, start=True
        )
        self._handle_camera_recording(
            "torso", self.teleop_gui.ui.torso_checkbox, start=True
        )
        self._handle_camera_recording(
            "hfg_left", self.teleop_gui.ui.hfg_left_checkbox, start=True
        )
        self._handle_camera_recording(
            "hfg_right", self.teleop_gui.ui.hfg_right_checkbox, start=True
        )
        self._handle_camera_recording(
            "robotiq", self.teleop_gui.ui.robotiq_checkbox, start=True
        )

    def _stop_svo_recordings(self):
        """
        Stops SVO recording for all selected cameras.

        Note: (Yash Shukla 06/04/2025):
            This method performs sequential service calls to each camera's stop_recording service.
            As the number of cameras grows, this can introduce a noticeable delay between the first
            and last camera's recording stop times. Consider threading these calls in the future to
            minimize per-episode startup latency and improve synchronization across cameras.
        """
        self._handle_camera_recording(
            "base", self.teleop_gui.ui.base_checkbox, start=False
        )
        self._handle_camera_recording(
            "torso", self.teleop_gui.ui.torso_checkbox, start=False
        )
        self._handle_camera_recording(
            "hfg_left", self.teleop_gui.ui.hfg_left_checkbox, start=False
        )
        self._handle_camera_recording(
            "hfg_right", self.teleop_gui.ui.hfg_right_checkbox, start=False
        )
        self._handle_camera_recording(
            "robotiq", self.teleop_gui.ui.robotiq_checkbox, start=False
        )

    def log(self, msg):
        print(msg)
        self.data_recording_log.emit(msg)
