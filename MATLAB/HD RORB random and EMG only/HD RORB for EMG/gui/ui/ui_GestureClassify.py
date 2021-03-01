# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/boardcontrol.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_GestureClassify(object):
    def setupUi(self, GestureClassify):
        GestureClassify.setObjectName(_fromUtf8("GestureClassify"))
        GestureClassify.resize(455, 127)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(GestureClassify.sizePolicy().hasHeightForWidth())
        GestureClassify.setSizePolicy(sizePolicy)

        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.gridLayout = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))

        # self.disconnBtn = QtGui.QPushButton(self.dockWidgetContents)
        # self.disconnBtn.setEnabled(False)
        # self.disconnBtn.setObjectName(_fromUtf8("disconnBtn"))
        # self.gridLayout.addWidget(self.disconnBtn, 3, 2, 1, 1)

        # self.refreshBtn = QtGui.QPushButton(self.dockWidgetContents)
        # self.refreshBtn.setObjectName(_fromUtf8("refreshBtn"))
        # self.gridLayout.addWidget(self.refreshBtn, 3, 0, 1, 1)

        # self.connectBtn = QtGui.QPushButton(self.dockWidgetContents)
        # self.connectBtn.setObjectName(_fromUtf8("connectBtn"))
        # self.gridLayout.addWidget(self.connectBtn, 3, 1, 1, 1)

        # self.label = QtGui.QLabel(self.dockWidgetContents)
        # self.label.setObjectName(_fromUtf8("label"))
        # self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        # self.selectBox = QtGui.QComboBox(self.dockWidgetContents)
        # self.selectBox.setObjectName(_fromUtf8("selectBox"))
        # self.gridLayout.addWidget(self.selectBox, 1, 1, 1, 2)
        
        GestureClassify.setWidget(self.dockWidgetContents)

        self.retranslateUi(GestureClassify)
        QtCore.QMetaObject.connectSlotsByName(GestureClassify)

    def retranslateUi(self, GestureClassify):
        GestureClassify.setWindowTitle(_translate("GestureClassify", "EMG Gesture Classification", None))
        # self.disconnBtn.setText(_translate("Experiment", "Disconnect", None))
        # self.refreshBtn.setText(_translate("Experiment", "Refresh", None))
        # self.connectBtn.setText(_translate("Experiment", "Connect", None))
        # self.label.setText(_translate("Experiment", "Connected control modules:", None))

