from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui.ui_GestureTraining import Ui_GestureTraining

class GestureTraining(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_GestureTraining()
        self.ui.setupUi(self)

    def setWorker(self, worker):
        pass