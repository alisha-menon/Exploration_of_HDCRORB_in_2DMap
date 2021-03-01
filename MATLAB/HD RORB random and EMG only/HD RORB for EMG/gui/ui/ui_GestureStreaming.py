# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/boardcontrol.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import os,sys

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_GestureStreaming(object):
    def setupUi(self, GestureStreaming):
        GestureStreaming.setObjectName(_fromUtf8("GestureStreaming"))

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(GestureStreaming.sizePolicy().hasHeightForWidth())
        GestureStreaming.setSizePolicy(sizePolicy)

        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.gridLayout = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))

        self.image = QtGui.QLabel(self.dockWidgetContents)
        self.image.setObjectName(_fromUtf8("image"))
        self.gridLayout.addWidget(self.image, 1, 0, 1, 6)

        self.gestureSets = QtGui.QComboBox(self.dockWidgetContents)
        self.gestureSets.setObjectName(_fromUtf8("gestureSets"))
        self.gridLayout.addWidget(self.gestureSets, 3, 0, 1, 2)

        self.numReps = QtGui.QSpinBox(self.dockWidgetContents)
        self.numReps.setMinimum(1)
        self.numReps.setMaximum(20)
        self.numReps.setSingleStep(1)
        self.numReps.setProperty("value", 5)
        self.numReps.setObjectName(_fromUtf8("numReps"))
        self.gridLayout.addWidget(self.numReps, 3, 2, 1, 1)

        self.timeGest = QtGui.QSpinBox(self.dockWidgetContents)
        self.timeGest.setMinimum(0)
        self.timeGest.setMaximum(10)
        self.timeGest.setSingleStep(1)
        self.timeGest.setProperty("value", 4)
        self.timeGest.setObjectName(_fromUtf8("timeGest"))
        self.gridLayout.addWidget(self.timeGest, 3, 3, 1, 1)

        self.timeRelax = QtGui.QSpinBox(self.dockWidgetContents)
        self.timeRelax.setMinimum(0)
        self.timeRelax.setMaximum(10)
        self.timeRelax.setSingleStep(1)
        self.timeRelax.setProperty("value", 3)
        self.timeRelax.setObjectName(_fromUtf8("timeRelax"))
        self.gridLayout.addWidget(self.timeRelax, 3, 4, 1, 1)

        self.timeTrans = QtGui.QSpinBox(self.dockWidgetContents)
        self.timeTrans.setMinimum(0)
        self.timeTrans.setMaximum(10)
        self.timeTrans.setSingleStep(1)
        self.timeTrans.setProperty("value", 2)
        self.timeTrans.setObjectName(_fromUtf8("timeTrans"))
        self.gridLayout.addWidget(self.timeTrans, 3, 5, 1, 1)

        self.label1 = QtGui.QLabel(self.dockWidgetContents)
        self.label1.setObjectName(_fromUtf8("label1"))
        self.gridLayout.addWidget(self.label1, 2, 0, 1, 1)

        self.label2 = QtGui.QLabel(self.dockWidgetContents)
        self.label2.setObjectName(_fromUtf8("label2"))
        self.gridLayout.addWidget(self.label2, 2, 2, 1, 1)

        self.label3 = QtGui.QLabel(self.dockWidgetContents)
        self.label3.setObjectName(_fromUtf8("label3"))
        self.gridLayout.addWidget(self.label3, 2, 3, 1, 1)

        self.label4 = QtGui.QLabel(self.dockWidgetContents)
        self.label4.setObjectName(_fromUtf8("label4"))
        self.gridLayout.addWidget(self.label4, 2, 4, 1, 1)

        self.label5 = QtGui.QLabel(self.dockWidgetContents)
        self.label5.setObjectName(_fromUtf8("label5"))
        self.gridLayout.addWidget(self.label5, 2, 5, 1, 1)

        self.subInd = QtGui.QSpinBox(self.dockWidgetContents)
        self.subInd.setMinimum(1)
        self.subInd.setMaximum(100)
        self.subInd.setSingleStep(1)
        self.subInd.setProperty("value", 1)
        self.subInd.setObjectName(_fromUtf8("subInd"))
        self.gridLayout.addWidget(self.subInd, 4, 1, 1, 1)

        self.label6 = QtGui.QLabel(self.dockWidgetContents)
        self.label6.setObjectName(_fromUtf8("label6"))
        self.gridLayout.addWidget(self.label6, 4, 0, 1, 1)

        self.expInd = QtGui.QSpinBox(self.dockWidgetContents)
        self.expInd.setMinimum(1)
        self.expInd.setMaximum(100)
        self.expInd.setSingleStep(1)
        self.expInd.setProperty("value", 1)
        self.expInd.setObjectName(_fromUtf8("expInd"))
        self.gridLayout.addWidget(self.expInd, 4, 3, 1, 1)

        self.label7 = QtGui.QLabel(self.dockWidgetContents)
        self.label7.setObjectName(_fromUtf8("label7"))
        self.gridLayout.addWidget(self.label7, 4, 2, 1, 1)

        self.saveData = QtGui.QCheckBox(self.dockWidgetContents)
        self.saveData.setObjectName(_fromUtf8("saveData"))
        self.gridLayout.addWidget(self.saveData, 4, 4, 1, 1)
        self.saveData.setChecked(True)

        self.wideDisable = QtGui.QPushButton(self.dockWidgetContents)
        self.wideDisable.setObjectName(_fromUtf8("wideDisable"))
        self.gridLayout.addWidget(self.wideDisable, 4, 5, 1, 1)

        self.description = QtGui.QLineEdit(self.dockWidgetContents)
        self.description.setObjectName(_fromUtf8("description"))
        self.gridLayout.addWidget(self.description, 5, 1, 1, 5)

        self.label8 = QtGui.QLabel(self.dockWidgetContents)
        self.label8.setObjectName(_fromUtf8("label8"))
        self.gridLayout.addWidget(self.label8, 5, 0, 1, 1)

        self.message = QtGui.QLabel(self.dockWidgetContents)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(40)
        self.message.setFont(font)
        self.message.setAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.message.setObjectName(_fromUtf8("message"))
        self.gridLayout.addWidget(self.message, 0, 0, 1, 6)
        
        GestureStreaming.setWidget(self.dockWidgetContents)
        self.retranslateUi(GestureStreaming)
        QtCore.QMetaObject.connectSlotsByName(GestureStreaming)

    def retranslateUi(self, GestureStreaming):
        GestureStreaming.setWindowTitle(_translate("GestureStreaming", "EMG Gesture Streaming", None))
        self.label1.setText(_translate("Experiment", "Gesture Set:", None))
        self.label2.setText(_translate("Experiment", "Repetitions:", None))
        self.label3.setText(_translate("Experiment", "Gesture Length (s):", None))
        self.label4.setText(_translate("Experiment", "Relax Length (s):", None))
        self.label5.setText(_translate("Experiment", "Transition Length (s):", None))
        self.label6.setText(_translate("Experiment", "Subject:", None))
        self.label7.setText(_translate("Experiment", "Experiment #:", None))
        self.label8.setText(_translate("Experiment", "Description:", None))
        self.saveData.setText(_translate("Experiment", "Save Data", None))
        self.message.setText("Current gesture\nNext gesture in 5")
        self.wideDisable.setText(_translate("Experiment", "Disable Wide In", None))

