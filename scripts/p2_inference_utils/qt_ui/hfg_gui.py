from PyQt5.QtCore import QObject

from bg_high_flow_valve.high_flow_valve_jvl import HighFlowValveJVL


class HighFlowGripperGui(QObject):
    def __init__(self, parent: QObject, ui, valve: HighFlowValveJVL | None):
        super().__init__(parent)
        self.ui = ui
        self.valve = valve
        self.ui.send_valve_position_button.clicked.connect(self.send_valve_position)
        self.ui.valve_position_slider.valueChanged.connect(
            self.update_valve_position_label
        )

    def update_valve_position_label(self, value: int):
        valve_position = value / self.ui.valve_position_slider.maximum()
        self.ui.valve_position_label.setText(f"{valve_position:.2f}")

    def send_valve_position(self):
        valve_position = (
            self.ui.valve_position_slider.value()
            / self.ui.valve_position_slider.maximum()
        )
        print(f"Valve position: {valve_position}")
        if self.valve is not None:
            self.valve.command_position(position=valve_position, slew_rate_scale=1.0)
        else:
            print("Valve is None!")
