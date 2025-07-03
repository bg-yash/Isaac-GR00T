from PyQt5.QtCore import (
    QTimer,
    Qt,
)
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QStyle,
    QLabel,
    QPushButton,
    QFrame,
)


class QToastNotification(QFrame):
    def __init__(self, parent, message, timeout_ms=5000):
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)
        self.setWindowOpacity(0.9)

        self.initUI(message)

        QTimer.singleShot(timeout_ms, self.close)

        # Move the notification to the top left corner of the parent window
        self.move(parent.x() + 100, parent.y() + 100)

    def initUI(self, message):
        layout = QHBoxLayout()

        # Message label
        message_label = QLabel(message)
        layout.addWidget(message_label)

        close_button = QPushButton()
        close_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        close_button.clicked.connect(self.close)

        layout.addWidget(close_button)

        self.setLayout(layout)

        self.setStyleSheet(
            """
            QFrame {
                background-color: #aaa661;
            }
            QPushButton {
                border: none;
            }
            """
        )
