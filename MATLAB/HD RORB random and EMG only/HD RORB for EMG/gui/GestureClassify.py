from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui.ui_GestureClassify import Ui_GestureClassify
import numpy as np
# import hdc
# import HD_model

class GestureClassify(QDockWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.ui = Ui_GestureClassify()
		self.ui.setupUi(self)
		self.windowSize = 1000
		self.classifyPeriod = 500
		self.numTicks = self.classifyPeriod/50
		self.tickCount = 0
		self.numElectrodes = 64
		self.dataWindow = np.zeros((self.windowSize,self.numElectrodes))

	def setWorker(self, worker):
		pass

	@pyqtSlot(list)
	def tick(self, data):
		if data:
			datalen = len(data)
			self.dataWindow = np.delete(self.dataWindow,range(datalen),0)
			self.dataWindow = np.append(self.dataWindow,np.asarray(data),0)
			self.tickCount = (self.tickCount + 1)%self.numTicks
			if self.tickCount == 0:
				self.classify()

	def classify(self):
		rms = np.std(self.dataWindow,0)
		print(rms)

		# compute n-gram
		