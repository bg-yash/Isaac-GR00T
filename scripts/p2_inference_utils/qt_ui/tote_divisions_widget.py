"""
Custom Qt Widget for selecting tote divisions.
A grid layout is used and QPushButtons are added to the grid.
We assume the name of the tote contains the division information as "{n}r" or "{n}c" with a leading underscore.
If no such format is found, we assume there is only 1 division.
"""

import re

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QButtonGroup,
    QRadioButton,
)


class ToteDivisionsWidget(QWidget):
    tote_selected = pyqtSignal(str, int, int)

    def __init__(self, parent, parent_layout):
        super().__init__(parent)
        self.parent_layout = parent_layout

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(0)
        self.grid.setVerticalSpacing(0)

        self.parent_layout.addLayout(self.grid)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

    def update_divisions(self, tote_name: str):
        for button in self.button_group.buttons():
            self.button_group.removeButton(button)
            button.deleteLater()

        # Check if the tote name contains the division information
        rc_with_leading_underscore_match = re.search(
            r"_(?=\d+[rc])(\d+r)?(\d+c)?", tote_name
        )
        if not rc_with_leading_underscore_match:
            rows, cols = 1, 1
        else:
            row_group, col_group = rc_with_leading_underscore_match.groups()
            rows = 1 if row_group is None else int(row_group[:-1])
            cols = 1 if col_group is None else int(col_group[:-1])

        idx = 0
        # We go backwards here in order to match the convention used by RPC
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                button_text = f"division_{idx}"
                radio_button = QRadioButton(button_text, self)
                radio_button.setStyleSheet(
                    "QRadioButton { border: 2px solid #555522; border-radius: 3px; padding: 1px; }"
                )
                radio_button.toggled.connect(
                    lambda checked: self.tote_selected.emit(
                        f"{tote_name}_{button_text}", r, c
                    )
                )
                self.button_group.addButton(radio_button)
                self.grid.addWidget(radio_button, r, c)

                idx += 1

    def get_focus_proxy(self):
        if self.button_group.buttons():
            return self.button_group.buttons()[0]
        else:
            return None

    def get_selection(self):
        for button in self.button_group.buttons():
            if button.isChecked():
                return button.text()
        return ""

    def set_selection(self, division_name: str):
        for button in self.button_group.buttons():
            if button.text() == division_name:
                button.setChecked(True)
                return
