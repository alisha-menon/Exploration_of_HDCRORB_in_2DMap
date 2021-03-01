from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui.ui_GestureStreaming import Ui_GestureStreaming
import os
import tables
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from gestureDef import gestureNames, gestureGroupNames, gestureGroupMembers

class GestureStreaming(QDockWidget):
    
    wideDisable = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_GestureStreaming()
        self.ui.setupUi(self)

        self.gestureList = []
        for key, value in gestureGroupNames.items():
            self.gestureList.append(value)
        self.ui.gestureSets.addItems(self.gestureList)

        self.messages = []
        self.numMessages = 0
        self.messageIdx = 0

        self.images = []
        self.imageDir = os.getcwd() + "/Gestures/"

        self.ui.wideDisable.clicked.connect(self.emitWideDisable)

    def setWorker(self, worker):
        pass

    def initImage(self):
    	self.ui.image.setPixmap(QPixmap(self.imageDir + "Rest.png").scaledToWidth(self.ui.image.geometry().width()))

    @pyqtSlot()
    def start(self):
        self.gestGroup = self.ui.gestureSets.currentIndex()
        self.gestGroupName = gestureGroupNames[self.gestGroup]
        self.gestLabels = gestureGroupMembers[self.gestGroup]
        self.gestNames = []
        for label in self.gestLabels:
            self.gestNames.append(gestureNames[label])

        self.numGest = len(self.gestNames)
        self.reps = self.ui.numReps.value()
        self.gestSecs = self.ui.timeGest.value()
        self.transSecs = self.ui.timeTrans.value()
        self.relaxSecs = self.ui.timeRelax.value()

        self.bufferRelaxSecs = 5
        # self.numMessages = (2*bufferRelaxSecs) + (numGest*reps*gestSecs) + (numGest*reps*transSecs*2) + ((numGest*reps - 1)*relaxSecs)

        # build sequence of messages
        self.messages = []
        self.images = []
        # fill in start messages
        for x in range(self.bufferRelaxSecs,0,-1):
            self.messages.append('Relax\nBegin with ' + self.gestNames[0] + ' in ' + str(x))
            self.images.append('Rest')

        for i,g in enumerate(self.gestNames):
            for x in range(1,self.reps+1):
                # reach to gesture
                for s in range(self.transSecs,0,-1):
                    self.messages.append('Reach ' + g + ' in ' + str(s) + ' seconds \nTrial #' + str(x))
                    self.images.append(g.replace(" ",""))
                # hold gesture
                for s in range(self.gestSecs,0,-1):
                    self.messages.append('Hold ' + g + ' for ' + str(s) + ' seconds \nTrial #' + str(x))
                    self.images.append(g.replace(" ",""))
                # relax gesture
                for s in range(self.transSecs,0,-1):
                    self.messages.append('Relax ' + g + ' in ' + str(s) + ' seconds \nTrial #' + str(x))
                    self.images.append('Rest')
                # relax in between gestures
                if x != self.reps:
                    for s in range(self.relaxSecs,0,-1):
                        self.messages.append('Relax\n' + g + ' in ' + str(s) + ' seconds')
                        self.images.append('Rest')
                else:
                    if i != len(self.gestNames)-1:
                        for s in range(self.relaxSecs,0,-1):
                            self.messages.append('Relax\n' + self.gestNames[i+1] + ' in ' + str(s) + ' seconds')
                            self.images.append('Rest')

        # fill in end messages
        for x in range(self.bufferRelaxSecs,0,-1):
            self.messages.append('Relax\nDone in ' + str(x))
            self.images.append('Rest')
        self.messages.append('Done!\n')
        self.images.append('Rest')

        self.numMessages = len(self.messages)
        print(self.numMessages)
        self.ui.message.setText('Experiment Begin\n' + self.gestGroupName)
        self.messageIdx = 0
        self.ui.image.setPixmap(QPixmap(self.imageDir + "Rest.png").scaledToWidth(self.ui.image.geometry().width()))

    @pyqtSlot()
    def tick(self):
        self.ui.image.setPixmap(QPixmap(self.imageDir + self.images[self.messageIdx] + ".png").scaledToWidth(self.ui.image.geometry().width()))
        self.ui.message.setText(self.messages[self.messageIdx])
        if self.messageIdx < self.numMessages - 1:
            self.messageIdx += 1

    @pyqtSlot(str)
    def stop(self,file):
        if self.ui.saveData.isChecked():
            start = file.find('hdfs/') + 5
            end = file.find('.hdf',start)
            timeStamp = file[start:end]

            hdfFile = tables.open_file(file, mode='r')
            dataTable = hdfFile.root.dataGroup.dataTable
            raw = np.asarray([x['out'] for x in dataTable.iterrows()])
            
            # crcs
            crc = raw[:,0]
            raw = raw[:,1:97]

            for i,s in enumerate(crc):
                if s==0xff and i!=0:
                    for ch in range(96):
                        raw[i,ch] = raw[i-1,ch]


            # plt.plot(raw)
            # plt.show()

            subInd = self.ui.subInd.value()
            expInd = self.ui.expInd.value()

            saveDir = 'data/mat/' + str(subInd).zfill(3) + '/' + str(expInd) + '/' + timeStamp + '/'
            os.makedirs(saveDir, exist_ok=True)

            trialLen = self.gestSecs + 2*self.transSecs + self.relaxSecs
            print('trialLen: ' + str(trialLen))
            if len(raw) >= self.numMessages*1000:
                # create matlab file for each repetition of each gesture
                for i,g in enumerate(self.gestLabels):
                    for x in range(self.reps):
                        fileStart = (self.bufferRelaxSecs + 1 - self.relaxSecs/2 + x*trialLen + i*self.reps*trialLen)*1000
                        fileEnd = fileStart + trialLen*1000
                        gestStart = (self.relaxSecs/2)*1000
                        gestEnd = gestStart + (self.gestSecs + 2*self.transSecs)*1000

                        print(str(fileStart) + ' ' + str(fileEnd) + ' ' + str(gestStart) + ' ' + str(gestEnd))

                        # create data
                        data = raw[int(fileStart):int(fileEnd),:]
                        # create gesture label with correct value
                        gestLabel = np.zeros(trialLen*1000)
                        gestLabel[int(gestStart):int(gestEnd)] = g
                        # create info struct
                        streamInfo = {'subject': subInd, 'description': str(self.ui.description.text()), 'rep': x+1, 'timeGest': self.gestSecs*1000, 'timeRelax': self.relaxSecs*1000, 'timeTrans': self.transSecs*1000, 'lsbmV': 0.0031, 'bufferRelaxSecs': self.bufferRelaxSecs*1000, 'timeStamp': timeStamp, 'gestNum': g}
                        # matfile = 'data/mat/' + str(subInd).zfill(3) + '_' + str(g).zfill(3) + '_' + timeStamp + '_' + str(x+1) + '.mat'
                        matfile = saveDir + str(subInd).zfill(3) + '_' + str(expInd) + '_' + str(g).zfill(3) + '_' + timeStamp + '_' + str(x+1) + '.mat'
                        sio.savemat(matfile, {'data':data, 'gestLabel':gestLabel, 'streamInfo':streamInfo})
            hdfFile.close()
        else:
            os.remove(file)

    @pyqtSlot()
    def emitWideDisable(self):
        self.wideDisable.emit()

