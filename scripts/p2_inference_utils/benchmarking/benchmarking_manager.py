import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from uuid import uuid4

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication

from p2_teleop.benchmarking.analysis import episode_success
from p2_teleop.benchmarking.db_utils import (
    get_sku_db_collection,
    get_sku_id_by_product_name,
)
from p2_teleop.constants import TransferTermination
from p2_teleop.episode_recording.episode_utils import load_episode
from p2_teleop.filesystem_utils import mkdir_as_logs_group
from p2_teleop.qt_ui.dialogs import StartEpisodeDialog
from p2_teleop.qt_ui.teleop_gui import TeleopMainWindow


class BenchmarkingManager:
    def __init__(self, teleop_gui: TeleopMainWindow, root: Path):
        """

        :param teleop_gui:
        :param root: For example, '/data'
        """
        self.teleop_gui = teleop_gui
        # Used for naming
        self.first_episode_data = None
        self.is_benchmarking = False
        self.root = root
        self.data = {}
        self.reset_data()

        self.skus = get_sku_db_collection()

    def on_episode_start(self, start_episode_data: Dict, episode_path: Path):
        if not self.is_benchmarking:
            return

        self.first_episode_data = start_episode_data

        self.data["episode_paths"].append(str(episode_path))
        self.save()

    def on_episode_end(self):
        if not self.is_benchmarking:
            return

        n_completed_transfers = len(self.data["episode_paths"])
        n_intended_transfers = self.data["intended_transfers_count"]
        n_succeeded = self.get_n_succeeded()

        self.teleop_gui.n_succeeded_edit.setText(f"{n_succeeded}")
        self.teleop_gui.n_completed_edit.setText(f"{n_completed_transfers}")
        self.teleop_gui.n_intended_total_edit.setText(f"{n_intended_transfers}")

        color_scheme = QGuiApplication.instance().styleHints().colorScheme()
        if color_scheme == Qt.ColorScheme.Dark:
            default_text_color = "white"
        else:
            default_text_color = "black"

        # set the text color to highlight when the number of completed transfers is less than the
        # intended transfers
        if n_succeeded == 0:
            n_succeeded_text_color = default_text_color
        elif n_succeeded < n_intended_transfers:
            n_succeeded_text_color = "yellow"
        elif n_succeeded == n_intended_transfers:
            n_succeeded_text_color = "green"
        else:
            # Tote should automatically end here.
            n_succeeded_text_color = "orange"

        if n_completed_transfers <= n_intended_transfers:
            n_completed_transfers_text_color = default_text_color
        else:
            n_completed_transfers_text_color = "orange"

        self.teleop_gui.n_succeeded_edit.setStyleSheet(
            f"color: {n_succeeded_text_color}"
        )
        self.teleop_gui.n_completed_edit.setStyleSheet(
            f"color: {n_completed_transfers_text_color}"
        )

        if self.is_tote_successfully_completed():
            self.teleop_gui.open_end_tote_dialog()

    def on_tote_start(self, tote_start_data: Dict):
        """

        :param tote_start_data: Data from the StartToteDialog.
        :return:
        """
        self.is_benchmarking = True
        self.reset_data()
        self.data.update(tote_start_data)

        intended_transfers = self.data["intended_transfers_count"]
        self.teleop_gui.n_succeeded_edit.setText("0")
        self.teleop_gui.n_completed_edit.setText("0")
        self.teleop_gui.n_intended_total_edit.setText(f"{intended_transfers}")

    def is_tote_successfully_completed(self):
        """
        Checks whether the tote is complete, which is used to automatically end the tote.
        We currently only do this if we have the right number of successful transfers.
        """
        # open all the episode metadata files and check for success
        n_success = self.get_n_succeeded()
        return n_success == self.data["intended_transfers_count"]

    def get_n_succeeded(self):
        n_success = 0
        for episode_path in self.data["episode_paths"]:
            episode = load_episode(episode_path)
            if episode_success(episode):
                n_success += 1
        return n_success

    def end_episode_as_success(self):
        end_data = {
            "transfer_termination": TransferTermination.SUCCESS.name,
        }
        self.teleop_gui.episode_ended.emit(end_data)

    def repeat_start_next_episode(self):
        # If there are more episodes needed for this tote, start the next one
        if not self.is_tote_successfully_completed():
            start_dialog = StartEpisodeDialog(self.teleop_gui)
            start_dialog.restore_settings()
            start_data = start_dialog.get_data()

            self.teleop_gui.episode_started.emit(start_data)
            self.teleop_gui.update_metadata(start_data)

    def reset_data(self):
        self.data = {
            "episode_paths": [],
            "uuid": str(uuid4()),
        }

    def on_tote_end(self, data: Dict):
        """

        :param data: Data from the EndToteDialog.
        :return:
        """
        self.is_benchmarking = False
        self.data.update(data)
        self.data["n_transfers_completed"] = len(self.data["episode_paths"])
        self.save()

        self.teleop_gui.n_succeeded_edit.setText("")
        self.teleop_gui.n_completed_edit.setText("")
        self.teleop_gui.n_intended_total_edit.setText("")

    def save(self):
        root = (
            self.root / "benchmarking_data" / f"{datetime.now().strftime('%Y-%m-%d')}"
        )
        mkdir_as_logs_group(root)

        product_id = self.first_episode_data["product_id"]
        product_name = self.skus[product_id]
        identifier = "_".join(
            [
                datetime.now().strftime("%Y-%m-%d"),
                product_name,
                self.first_episode_data["gripper_id"],
                self.first_episode_data["tote_type"],
                self.first_episode_data["division_id"],
            ]
        )
        in_progress_save_path = (
            root / f"{identifier}_benchmarking_{self.data['uuid']}_in_progress.json"
        )
        complete_save_path = (
            root / f"{identifier}_benchmarking_{self.data['uuid']}.json"
        )
        if self.is_benchmarking:
            with in_progress_save_path.open("w") as f:
                json.dump(self.data, f, indent=2)
        else:
            in_progress_save_path.unlink(missing_ok=True)
            with complete_save_path.open("w") as f:
                json.dump(self.data, f, indent=2)
