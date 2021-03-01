from PyQt4.QtCore import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from ui.ui_DataVisualizer import Ui_DataVisualizer
import numpy as np
import scipy.io
from enum import Enum
import pyqtgraph as pg
import tables
from tables import *
from datetime import datetime
import time
# import QThread
import cmbackend
import csv
from scipy import signal

class DataVisualizer(QDockWidget):
    streamAdc = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_DataVisualizer()
        self.ui.setupUi(self)

        self.data = []
        self.numPlots = 4
        self.xRange = self.ui.xRange.value() # number of ms (samples) over which to plot continuous data

        self.dataPlot = np.zeros((self.numPlots, self.ui.xRange.maximum())) # aggregation of data to plot (scrolling style)
        self.plotPointer = 0 # pointer to current x position in plot (for plotting scrolling style)

        self.plots = [] # stores pyqtgraph objects
        self.plotColors = []
        self.target = 50


        self.plotCh = np.arange(self.numPlots)

        # initialize streaming mode thread
        self.streamAdcThread = cmbackend.streamAdcThread()
        self.streamAdcThread.streamAdcData.connect(self.streamAdcData)
        # self.connect(self.streamAdcThread, SIGNAL('streamDataOut(PyQt_PyObject)'), self.streamAdcData)

        self.setWindowTitle("Data Visualizer")

        # populate arrays
        for i in range(0,self.numPlots):
            self.plots.append(pg.PlotItem().plot())
            self.plotColors.append(pg.intColor(i%16,16))

        self.updatePlotDisplay()

        self.ui.autorange.clicked.connect(self.updatePlotStream)
        self.ui.xRange.valueChanged.connect(self.updatePlotDisplay)
        self.ui.clearBtn.clicked.connect(self.clearPlots)

        self.ui.ch0.valueChanged.connect(self.updateCh)
        self.ui.ch1.valueChanged.connect(self.updateCh)
        self.ui.ch2.valueChanged.connect(self.updateCh)
        self.ui.ch3.valueChanged.connect(self.updateCh)
        self.ui.target.valueChanged.connect(self.updateTarget)

        # set some defaults
        self.ui.autorange.setChecked(True)

        self.raw = np.zeros((200,64))

    @pyqtSlot()
    def updateCh(self):
        self.plotCh[0] = self.ui.ch0.value()
        self.plotCh[1] = self.ui.ch1.value()
        self.plotCh[2] = self.ui.ch2.value()
        self.plotCh[3] = self.ui.ch3.value()

    @pyqtSlot()
    def updateTarget(self):
        self.target = self.ui.target.value()
        print(self.target)

    @pyqtSlot(list)
    def streamAdcData(self, data):
        if data:
            self.data = data
            if self.ui.dispStream.isChecked():
                self.updatePlotStream()
            
            # UPDATE EFFORT LEVEL BAR HERE!
            numSamples = len(data)
            self.raw = np.delete(self.raw, slice(0,numSamples), 0)
            self.raw = np.append(self.raw, np.asarray(data), axis=0)
            effort = np.mean(np.std(signal.detrend(self.raw, axis=0),axis=0))
            effort = int((effort + self.ui.b.value())*self.ui.m.value())
            effort = max(min(self.ui.effort.maximum(), effort), self.ui.effort.minimum())
            self.ui.effort.setValue(effort)

    @pyqtSlot()
    def on_streamBtn_clicked(self):
        if self.ui.streamBtn.isChecked():
            # self.streamAdcThread.setup(self.ui.dispStream.isChecked())
            self.streamAdcThread.setup(True, self.ui.classify.isChecked())
            self.streamAdcThread.start()
        else:
            filename = self.streamAdcThread.stop()

    @pyqtSlot()
    def clearPlots(self):
        self.plotPointer = 0
        self.dataPlot = np.zeros(
            (self.numPlots, self.ui.xRange.maximum()))  # aggregation of data to plot (scrolling style)
        for ch in range(0, self.numPlots):
            self.plots[ch].clear()

        effort = self.ui.effort.value()


    @pyqtSlot()
    def updatePlotDisplay(self):
        for j in reversed(range(0,self.numPlots)):
            plotToDelete = self.ui.plot.getItem(j,0)
            if plotToDelete is not None:
                self.ui.plot.removeItem(plotToDelete)

        self.xRange = self.ui.xRange.value()
        for i in range(0, self.numPlots):
            viewBox = pg.ViewBox(enableMouse=False)
            viewBox.setRange(xRange=[0,self.xRange])
            self.plots[i] = self.ui.plot.addPlot(row=i, col=0, viewBox=viewBox)
            self.plots[i].setLabel('left', text="Ch {}".format(self.plotCh[i]))

        # need to also replot the data
        self.updatePlotStream()

    @pyqtSlot()
    def updatePlotStream(self):
        if self.data:
            # loop through samples
            for t in range(0, len(self.data)):
                if self.plotPointer == self.xRange:
                    self.plotPointer = 0
                # grab specific sample
                temp = self.data[t]

                for ch in range(0, self.numPlots):
                    self.dataPlot[ch][self.plotPointer] = temp[self.plotCh[ch]]
                
                self.plotPointer += 1
            self.data = []

        for ch in range(0, self.numPlots):
            dp = self.dataPlot[ch][0:self.xRange]

            avg = np.mean(dp)
            sd = np.std(dp)
            if sd < 10:
                sd = 10

            self.plots[ch].clear()
            self.plots[ch].plot(y=dp, pen=self.plotColors[ch])

            self.plots[ch].getViewBox().setMouseEnabled(x=True, y=True)
            self.plots[ch].getViewBox().setMouseMode(self.plots[ch].getViewBox().RectMode)
            self.plots[ch].getViewBox().setLimits(xMin=0, xMax=self.xRange, yMin=-32768, yMax=32768)
            self.plots[ch].getViewBox().setRange(yRange=(avg-(sd*2),avg+(sd*2)),update=True)
            self.plots[ch].setLabel('left', text="Ch {}".format(self.plotCh[ch]))
            
            if self.ui.autorange.isChecked():
                self.plots[ch].getViewBox().autoRange()

