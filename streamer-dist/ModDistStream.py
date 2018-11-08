#!/home/bicer/.conda/envs/py36/bin/python

import argparse
import numpy as np
import zmq
import time
import os
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__), './local'))
import flatbuffers
import TraceSerializer
import tomopy as tp
import tracemq as tmq


def parse_arguments():
  parser = argparse.ArgumentParser( description='Data Acquisition Process')
  parser.add_argument('--synchronize_subscriber', action='store_true',
      help='Synchronizes this subscriber to publisher (publisher should wait for subscriptions)')
  parser.add_argument('--subscriber_hwm', type=int, default=10*1024, 
      help='Sets high water mark value for this subscriber.')
  parser.add_argument('--publisher_address', default=None,
      help='Remote publisher address')
  parser.add_argument('--publisher_rep_address',
      help='Remote publisher REP address for synchronization')

  # TQ communication
  parser.add_argument('--bindip', default=None, help='IP address to bind tmq)')
  parser.add_argument('--port', type=int, default=5560,
                          help='Port address to bind tmq')
  parser.add_argument('--beg_sinogram', type=int,
                          help='Starting sinogram for reconstruction')
  parser.add_argument('--num_sinograms', type=int,
                          help='Number of sinograms to reconstruct (rows)')
  parser.add_argument('--num_columns', type=int,
                          help='Number of columns (cols)')

  parser.add_argument('--degree_to_radian', action='store_true',
              help='Converts rotation information to radian.')
  parser.add_argument('--mlog', action='store_true',
              help='Takes the minus log of projection data (projection data is divided by 50000 also).')
  parser.add_argument('--uint16_to_float32', action='store_true',
              help='Converts uint16 image byte sequence to float32.')
  parser.add_argument('--normalize', action='store_true',
              help='Normalizes incoming projection data with previously received dark and flat field.')
  parser.add_argument('--remove_invalids', action='store_true',
              help='Removes invalid measurements from incoming projections, i.e. negatives, nans and infs.')
  parser.add_argument('--remove_stripes', action='store_true',
              help='Removes stripes using fourier-wavelet method (in tomopy).')

  return parser.parse_args()


def synchronize_subs(context, publisher_rep_address):
  sync_socket = context.socket(zmq.REQ)
  sync_socket.connect(publisher_rep_address)

  sync_socket.send(b'') # Send synchronization signal
  sync_socket.recv() # Receive reply
  

def main():
  args = parse_arguments()

  context = zmq.Context()

  # TQM setup
  tmq.init_tmq()
  # Handshake w. remote processes
  tmq.handshake(args.bindip, args.port, args.num_sinograms, args.num_columns)

  # Subscriber setup
  subscriber_socket = context.socket(zmq.SUB)
  subscriber_socket.set_hwm(args.subscriber_hwm)
  subscriber_socket.connect(args.publisher_address)
  subscriber_socket.setsockopt(zmq.SUBSCRIBE, b'')

  if args.synchronize_subscriber:
    synchronize_subs(context, args.publisher_rep_address)


  # Setup flatbuffer builder and serializer
  builder = flatbuffers.Builder(0)
  serializer = TraceSerializer.ImageSerializer(builder)

  # Receive images
  total_received=0
  total_size=0

  time0 = time.time()
  while True:
    msg = subscriber_socket.recv()
    if msg == b"end_data": break # End of data acquisition
    # Deserialize msg to image
    read_image = serializer.deserialize(serialized_image=msg)
    serializer.info(read_image) # print image information
    # Local checks
    seq=read_image.Seq()
    if seq!=total_received: 
      print("Wrong sequence number: {} != {}".format(seq, total_received))

    # Push image to workers (REQ/REP)
    my_image_np = read_image.TdataAsNumpy()
    if args.uint16_to_float32:
      my_image_np.dtype = np.uint16
      sub = np.array(my_image_np, dtype="float32")
    sub = sub.reshape((read_image.Dims().Y(), read_image.Dims().X()))
    #print(sub.shape)
    sub = sub[args.beg_sinogram:args.beg_sinogram+args.num_sinograms, :]



    # If incoming data is projection
    if read_image.Itype() is serializer.ITypes.Projection:
      rotation=read_image.Rotation()
      if args.degree_to_radian: rotation = rotation*math.pi/180.

      # Tomopy operations expect 3D data, reshape incoming projections.
      sub = np.reshape(sub, (1, sub.shape[0], sub.shape[1]))
      if args.normalize: sub = tp.normalize(sub, flat, dark) # XXX note flat/dark fields need to be the corresponding sinograms
      if args.remove_stripe: sub = tp.remove_stripe_fw(sub, level=7, wname='sym16', sigma=1, pad=True)
      if args.mlog: sub = -np.log(sub)
      if args.remove_nan:
        sub = tp.remove_nan(sub, val=0.00)
        sub = tp.remove_neg(sub, val=0.00)
        sub[np.where(sub == np.inf)] = 0.00

      ncols = sub.shape[2] 
      sub = sub.reshape(sub.shape[0]*sub.shape[1]*sub.shape[2])
      tmq.push_image(sub, args.num_sinograms, ncols, rotation, 
                      read_image.UniqueId(), read_image.Center())



    # If incoming data is white field
    if read_image.Itype() is serializer.ITypes.White: pass



    # If incoming data is white-reset
    if read_image.Itype() is serializer.ITypes.WhiteReset: pass



    # If incoming data is dark field
    if read_image.Itype() is serializer.ITypes.Dark: pass



    # If incoming data is dark-reset 
    if read_image.Itype() is serializer.ITypes.DarkReset: pass


    


    total_received += 1
    total_size += len(msg)
  time1 = time.time()
    
  # Profile information
  print("Received number of projections: {}".format(total_received))
  print("Rate = {} MB/sec; {} msg/sec".format(
            (total_size/(2**20))/(time1-time0), total_received/(time1-time0)))

  # Finalize TMQ
  tmq.done_image()
  tmq.finalize_tmq()




if __name__ == '__main__':
  main()
  
