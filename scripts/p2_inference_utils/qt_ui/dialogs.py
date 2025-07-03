from typing import Optional

from PyQt5 import uic
from PyQt5.QtCore import QSettings, QPropertyAnimation, QSequentialAnimationGroup
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QGraphicsColorizeEffect

from p2_teleop.benchmarking.db_utils import (
    get_sku_db_collection,
    get_gripper_ids,
    get_sku_id_by_product_name,
)
from p2_teleop.constants import UI_ROOT, TransferTermination, ToteTermination
from p2_teleop.qt_ui.fuzzy_combobox_search import ExtendedComboBox
from p2_teleop.qt_ui.tote_divisions_widget import ToteDivisionsWidget


def make_enum_combobox(combobox, enum_type, first_enum_item: Optional = None):
    # sort the names
    episode_termination_names = sorted([term.name for term in enum_type])
    # put "success" at the top
    if first_enum_item:
        episode_termination_names.remove(first_enum_item.name)
        episode_termination_names.insert(0, first_enum_item.name)

    # In the future we should probably sort these based on frequency in the existing data
    # make it searchable
    ebox = ExtendedComboBox(combobox)
    # Population the combobox based on the enum names
    for name in episode_termination_names:
        combobox.addItem(name)

    return ebox


class StartEpisodeDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = uic.loadUi(UI_ROOT / "start_episode_dialog.ui", self)

        self.tote_division_widget = ToteDivisionsWidget(
            self, self.ui.tote_divisions_layout
        )

        # initialize the tote divisions
        self.tote_division_widget.update_divisions(self.ui.tote_combobox.currentText())

        self.ui.tote_combobox.currentTextChanged.connect(
            self.tote_division_widget.update_divisions
        )

        self.ui.autofill_button.clicked.connect(self.autofill)

        # fill combobox options from database
        self.skus = get_sku_db_collection()
        for sku_name in self.skus.values():
            self.ui.product_name_combobox.addItem(sku_name)

        self.grippers = get_gripper_ids()
        for gripper_id in self.grippers:
            self.ui.gripper_id_combobox.addItem(gripper_id)

        # Clear so there's no default selection, the user must select one
        self.ui.product_name_combobox.setCurrentIndex(-1)
        self.ui.gripper_id_combobox.setCurrentIndex(-1)
        self.ui.pack_combobox.setCurrentIndex(-1)

        self.invalid_input_color_effect = QGraphicsColorizeEffect(
            self.ui.input_invalid_label
        )
        self.ui.input_invalid_label.setGraphicsEffect(self.invalid_input_color_effect)

        self.flash_color_anim1 = self.make_flash_color_anim(150)
        self.flash_color_anim2 = self.make_flash_color_anim(850)

        self.invalid_input_anim_group = QSequentialAnimationGroup(self)
        self.invalid_input_anim_group.addAnimation(self.flash_color_anim1)
        self.invalid_input_anim_group.addAnimation(self.flash_color_anim2)

    def make_flash_color_anim(self, duration):
        flash_color_anim = QPropertyAnimation(self.invalid_input_color_effect, b"color")
        flash_color_anim.setDuration(duration)
        flash_color_anim.setStartValue(QColor("white"))
        flash_color_anim.setEndValue(QColor("red"))
        return flash_color_anim

    def accept(self):
        """
        Override the accept method to check for required fields.

        :return:
        """
        self.save_settings()

        if self.ui.tote_combobox.currentText() == "":
            self.animate_invalid_input_label("Tote Type is required")
            return
        if self.tote_division_widget.get_selection() == "":
            self.animate_invalid_input_label("Division ID is required")
            return
        if self.ui.product_name_combobox.currentText() == "":
            self.animate_invalid_input_label("Product Name is required")
            return
        if self.ui.gripper_id_combobox.currentText() == "":
            self.animate_invalid_input_label("Gripper ID is required")
            return
        if self.ui.operator_id_combobox.currentText() == "":
            self.animate_invalid_input_label("Operator ID is required")
            return
        if self.ui.task_combobox.currentText() == "":
            self.animate_invalid_input_label("Task Type is required")
            return
        if self.ui.pack_combobox.currentText() == "":
            self.animate_invalid_input_label("Pack Type is required")
            return
        super().accept()

    def animate_invalid_input_label(self, message):
        """
        Set the input_invalid_label text and make the text color transition from white to red over 500ms.

        :param message:
        :return:
        """
        self.ui.input_invalid_label.setText(message)
        self.invalid_input_anim_group.start()

    def autofill(self):
        self.restore_settings()

    def get_data(self):
        curr_text = self.ui.product_name_combobox.currentText()
        product_id = get_sku_id_by_product_name(self.skus, curr_text)
        if product_id is None:
            # This could be a barcode that was scanned, so check that
            product_id = self.skus.get(curr_text, None)

        return {
            "tote_type": self.ui.tote_combobox.currentText(),
            "division_id": self.tote_division_widget.get_selection(),
            "product_id": product_id,
            "gripper_id": self.ui.gripper_id_combobox.currentText(),
            "operator_id": self.ui.operator_id_combobox.currentText(),
            "task_type": self.ui.task_combobox.currentText(),
            "pack_type": self.ui.pack_combobox.currentText(),
        }

    def save_settings(self):
        settings = self.get_settings()
        settings.setValue("product_name", self.ui.product_name_combobox.currentText())
        settings.setValue("tote_id", self.ui.tote_combobox.currentText())
        settings.setValue("division_id", self.tote_division_widget.get_selection())
        settings.setValue("gripper_id", self.ui.gripper_id_combobox.currentText())
        settings.setValue("operator_id", self.ui.operator_id_combobox.currentText())
        settings.setValue("task_type", self.ui.task_combobox.currentText())
        settings.setValue("pack_type", self.ui.pack_combobox.currentText())

    def restore_settings(self):
        settings = self.get_settings()
        if settings.contains("tote_id"):
            self.ui.tote_combobox.setCurrentText(settings.value("tote_id"))
        if settings.contains("division_id"):
            self.tote_division_widget.set_selection(settings.value("division_id"))
        if settings.contains("product_name"):
            self.ui.product_name_combobox.setCurrentText(settings.value("product_name"))
        if settings.contains("gripper_id"):
            self.ui.gripper_id_combobox.setCurrentText(settings.value("gripper_id"))
        if settings.contains("operator_id"):
            self.ui.operator_id_combobox.setCurrentText(settings.value("operator_id"))
        if settings.contains("task_type"):
            self.ui.task_combobox.setCurrentText(settings.value("task_type"))
        if settings.contains("pack_type"):
            self.ui.pack_combobox.setCurrentText(settings.value("pack_type"))

    def get_settings(self):
        settings = QSettings("BG", "P2Teleop")
        return settings


class EndEpisodeDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = uic.loadUi(UI_ROOT / "end_episode_dialog.ui", self)

        self.ebox = make_enum_combobox(
            self.ui.transfer_termination_combobox,
            TransferTermination,
            first_enum_item=TransferTermination.SUCCESS,
        )

    def get_data(self):
        # get the selected radio button from the group, and get the index
        difficulty_index = self.ui.difficulty_button_group.checkedId()
        return {
            "transfer_termination": self.ui.transfer_termination_combobox.currentText(),
            "perceived_difficulty": difficulty_index,
            "misc_notes": self.ui.misc_notes_edit.toPlainText(),
        }


class StartToteDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = uic.loadUi(UI_ROOT / "start_tote_dialog.ui", self)

    def get_data(self):
        return {
            "intended_transfers_count": self.ui.intended_transfers_spinbox.value(),
        }


class EndToteDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = uic.loadUi(UI_ROOT / "end_tote_dialog.ui", self)
        self.ebox = make_enum_combobox(
            self.ui.tote_termination_combobox,
            ToteTermination,
            first_enum_item=ToteTermination.SUCCESS,
        )

    def get_data(self):
        return {
            "tote_termination": self.ui.tote_termination_combobox.currentText(),
            "misc_notes": self.ui.misc_notes_edit.toPlainText(),
        }
