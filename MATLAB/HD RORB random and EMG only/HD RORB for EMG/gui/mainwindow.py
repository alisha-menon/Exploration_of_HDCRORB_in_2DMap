from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui.ui_mainwindow import Ui_MainWindow
from BoardControl import BoardControl
from cmdline import CommandLineWidget
from DataVisualizer import DataVisualizer
from GestureStreaming import GestureStreaming
from GestureClassify import GestureClassify

class MainWindow(QMainWindow):   
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.worker = None

        self.DataVisualizer = DataVisualizer(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.DataVisualizer)

        self.GestureStreaming = GestureStreaming(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.GestureStreaming)

        self.DataVisualizer.streamAdcThread.experimentTick.connect(self.GestureStreaming.tick)
        self.DataVisualizer.streamAdcThread.experimentStart.connect(self.GestureStreaming.start)
        self.DataVisualizer.streamAdcThread.experimentStop.connect(self.GestureStreaming.stop)

        self.GestureClassify = GestureClassify(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.GestureClassify)

        self.DataVisualizer.streamAdcThread.classifyTick.connect(self.GestureClassify.tick)

        self.tabifyDockWidget(self.GestureStreaming, self.GestureClassify)
        
        self.boardControl = BoardControl(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.boardControl)

        self.cmdline = CommandLineWidget(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.cmdline)

        self.tabifyDockWidget(self.boardControl, self.cmdline)

        s = QSettings()
        if s.value("mainwindow/geometry") is not None:
            self.restoreGeometry(s.value("mainwindow/geometry"))
            self.restoreState(s.value("mainwindow/state"))
        # run loadState when the event loop starts
        QTimer.singleShot(0, self.loadState)

    def setWorker(self, worker):
        self.worker = worker
        self.boardControl.setWorker(worker)
        self.GestureStreaming.wideDisable.connect(worker.wideDisable)

    @pyqtSlot()
    def loadState(self):
        s = QSettings()
        if s.value("mainwindow/state") is not None:
            self.restoreState(s.value("mainwindow/state"))

    def closeEvent(self, event):
        s = QSettings()
        s.setValue("mainwindow/geometry", self.saveGeometry())
        s.setValue("mainwindow/state", self.saveState())
        super().closeEvent(event)

