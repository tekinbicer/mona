#!/home/beams/USER2BMB/Apps/BlueSky/bin/python

import argparse
import sys
import time

import numpy as np
import pyqtgraph
from   pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from   pyqtgraph.widgets.RawImageWidget import RawImageWidget

import pvaccess
import tracemq as tmq
import h5py as h5
import math


framesDisplayed = 0
gain = None
black = None
MAX_FRAMES = 300

theta = 0.
start_sino = 64
num_sinos = 2


def setup_mock_data(input_f):
  ifptr = h5.File(input_f, 'r')
  global idata, itheta
  idata = np.array(ifptr['exchange/data'], dtype=np.float32)
  itheta = np.array(ifptr['exchange/theta'], dtype=np.float32)
  ifptr.close()


def mock_data_gen(proj_id):
  print('Proj={}/{}'.format(idata.shape[0], proj_id))
  if proj_id >= idata.shape[0]: return None, None
  return idata[proj_id], itheta[proj_id]



def update(v):
    global framesDisplayed, gain, black, MAX_FRAMES
    global start_sino, num_sinos, theta
    global mock_data

    i = v['value'][0]['ubyteValue']
    print(v)
    sys.exit(0)
    if not args.noAGC:
        if gain is None:
            # compute black point and gain from first frame
            black, white = np.percentile(i, [0.01, 99.99])
            gain = 255 / (white - black)
        i = (i - black) * gain
        i = np.clip(i, 0, 255).astype('uint8')

    # Push this image to remote processes (this is currently blocking)
    # Check if mock data
    if mock_data:
      sub, theta = mock_data_gen(framesDisplayed)
      if(sub is None): 
        app.quit()
        return
      if start_sino+num_sinos > sub.shape[0] : 
        print("Sinogram range exceeds!!")
        sys.exit(0)
      lx = sub.shape[1]
      sub = sub[start_sino:start_sino+num_sinos, :]
      sub = np.resize(sub, (sub.shape[0]*sub.shape[1]))
      theta = float(theta)
      print(theta)
      print(sub)
      center=1010.
    # Real data
    else:
      sub = i[start_sino*x : start_sino*x + num_sinos*x]
      theta = ((framesDisplayed%360)/180.)*math.pi
      lx = x
      center +=  0.001

    tmq.push_image(sub, num_sinos, lx, theta, framesDisplayed, center)
    framesDisplayed += 1

    # resize to get a 2D array from 1D data structure
    # i = np.resize(i, (y, x))
    #img.setImage(np.flipud(np.rot90(i)))
    #img.setImage(np.resize(sub, (num_sinos, lx)))
    app.processEvents()

    if args.benchmark and MAX_FRAMES>0 and framesDisplayed>MAX_FRAMES:
      app.quit()

def main():
    global app, img, x, y, MAX_FRAMES
    global args

    parser = argparse.ArgumentParser(
            description='AreaDetector video example')

    parser.add_argument("ImagePV",
                        help="EPICS PVA image PV name, such as 13PG2:Pva1:Image")
    parser.add_argument('--benchmark', action='store_true',
                        help='measure framerate')
    parser.add_argument('--noAGC', action='store_true',
                        help='disable auto gain')
    parser.add_argument('--frames',
                        help='maximum number of frames (-1: unlimited)')

    parser.add_argument('--bindip', default=None,
                        help='ip address to bind tmq)')
    parser.add_argument('--port', type=int, default=5560,
                        help='Port address to bind tmq')
    parser.add_argument('--beg_sinogram', type=int, 
                        help='Starting sinogram for reconstruction')
    parser.add_argument('--num_sinograms', type=int,
                        help='Number of sinograms to reconstruct')
    parser.add_argument('--mock_data', action='store_true', default=False,
                        help='Mock data acquisition from file')
    parser.add_argument('--mock_file',
                        help='File name for mock data acquisition')


    args = parser.parse_args()

    global start_sino, num_sinos, mock_data
    start_sino = args.beg_sinogram
    num_sinos = args.num_sinograms
    mock_data = args.mock_data

    if mock_data:
      print("Mock data is set") 
      setup_mock_data(args.mock_file)

    app = QtGui.QApplication([])
    win = QtWidgets.QWidget()
    win.setWindowTitle('daqScope')
    layout = QtGui.QGridLayout()
    layout.setMargin(0)
    win.setLayout(layout)
    img = RawImageWidget(win)
    layout.addWidget(img, 0, 0, 0, 0)
    win.show()
    chan = pvaccess.Channel(args.ImagePV)

    global x, y
    x,y = chan.get('field()')['dimension']
    x = x['size']
    y = y['size']
    win.resize(x, y)


    #### Setup TMQ #####
    tmq.init_tmq() 
    # Handshake w. remote processes
    # For set TMQ for 2 sinograms
    print(type(x))
    print(type(num_sinos))
    if mock_data:
      print("Mock data handshake: input data shape={}".format(idata.shape))
      tmq.handshake(args.bindip, args.port, int(num_sinos), idata.shape[2]) 
    else: 
      print("Streaming data handshake: input data shape={},{}".format(int(num_sinos), int(x)))
      tmq.handshake(args.bindip, args.port, int(num_sinos), int(x)) 
    ####################

    chan.subscribe('update', update)
    chan.startMonitor()

    if args.benchmark:
        start = time.time()

    # Finalize streaming ?
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        chan.stopMonitor()
        chan.unsubscribe('update')
        print("Remove")
        # Finalize TMQ
        tmq.done_image()
        tmq.finalize_tmq()

    if args.benchmark:
        stop = time.time()
        print('Frames displayed: %d' % framesDisplayed)
        print('Elapsed time:     %.3f sec' % (stop-start))
        print('Frames per second: %.3f FPS' % (framesDisplayed/(stop-start)))


if __name__ == '__main__':
    main()
