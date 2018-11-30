from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUi
import pyqtgraph.opengl as gl

import numpy as np
import MockVizData


class TSViewTrace(QWidget):
  def __init__(self):
    super(TSViewTrace, self).__init__()
    loadUi('TSViewTrace.ui', self)
    self.setWindowTitle('Trace Streaming Tomography Data Viewer')
    
    # Register your events below

    # buttonConnectPublisher
    self.buttonConnectPublisher.clicked.connect(self.buttonConnectPublisher_clicked)

    # buttonTestView
    self.buttonTestView.clicked.connect(self.buttonTestView_clicked)


  # buttonConnectPublisher events
  @pyqtSlot()
  def buttonConnectPublisher_clicked(self):
    self.labelStatusInfo.setText('Connecting {}'.format(self.textBoxPublisherAddress.text()))


  # buttonTestView events
  @pyqtSlot()
  def buttonTestView_clicked(self):
    ## Add grid to the 3D volume
    g = gl.GLGridItem()
    g.scale(10, 10, 1)
    self.graphicsViewStreamingData.addItem(g)

    ## Add data to the 3D volume
    data = MockVizData.Trace.generate()
    v = gl.GLVolumeItem(data)
    v.translate(-50,-50,-100)
    self.graphicsViewStreamingData.addItem(v)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tsView = TSViewTrace()
    tsView.show()
    sys.exit(app.exec_())

  
