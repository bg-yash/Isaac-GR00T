from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import QComboBox, QCompleter


class ExtendedComboBox:
    """Makes a combobox more easily searchable"""

    def __init__(self, combobox: QComboBox):
        self.box = combobox

        self.box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.box.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self.box)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.box.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self.box)
        # always show all (filtered) completions
        self.completer.setCompletionMode(
            QCompleter.CompletionMode.UnfilteredPopupCompletion
        )
        self.box.setCompleter(self.completer)

        # connect signals
        self.box.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    # on selection of an item from the completer, select the corresponding item from combobox
    def on_completer_activated(self, text):
        if text:
            index = self.box.findText(text)
            self.box.setCurrentIndex(index)
            self.box.activated[str].emit(self.box.itemText(index))

    # on model change, update the models of the filter and completer as well
    def setModel(self, model):
        self.box.setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        self.box.setModelColumn(column)
