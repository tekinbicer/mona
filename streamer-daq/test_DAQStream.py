#!/home/beams/USER2BMB/Apps/BlueSky/bin/python

import argparse
import sys
import numpy as np
import zmq
import TraceSerializer
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './local'))
import flatbuffers

def parse_arguments():
  parser = argparse.ArgumentParser(
          description='Data Acquisition Process')
  parser.add_argument("ImagePV",
                      help="EPICS PVA image PV name, such as 13PG2:Pva1:Image")

  parser.add_argument('--bind_address_publisher', default="tcp://*:5560",
                      help='Address to bind publisher')
  parser.add_argument('--publisher_hwm', type=int, default=10*1024,
                      help='Sets high water mark value for publisher.')
  parser.add_argument('--synchronize_subscribers', action='store_true',
                      help='Waits for all subscribers to join.')
  parser.add_argument('--subscriber_count', type=int,
                      help='Number of expected subscribers')
  parser.add_argument('--bind_address_rep', default=None,
                      help='Address to bind REP socket (for synchronization)')

  parser.add_argument('--beg_sinogram', type=int, 
                      help='Starting sinogram for reconstruction')
  parser.add_argument('--num_sinograms', type=int,
                      help='Number of sinograms to reconstruct')
  parser.add_argument('--simulate_daq', action='store_true', default=False,
                      help='Simulate data acquisition from file')
  parser.add_argument('--simulation_file',
                        help='File name for mock data acquisition')
  return parser.parse_args()

def synchronize_subs(context, subscriber_count, bind_address_rep):
  print("Synching")
  # Socket to receive signals
  sync_socket = context.socket(zmq.REP)
  sync_socket.bind(bind_address_rep)
  
  # Get synchronization from subscribers
  counter = 0
  while counter < subscriber_count:
    # wait for synchronization request
    msg = sync_socket.recv()
    # send synchronization reply
    sync_socket.send(b'')
    counter += 1
    print("Joined subscriber: {}/{}".format(counter, subscriber_count))


def setup_simulation_data(input_f):
  ifptr = h5.File(input_f, 'r')
  idata = np.array(ifptr['exchange/data'], dtype=np.float32)
  itheta = np.array(ifptr['exchange/theta'], dtype=np.float32)
  ifptr.close()
  return idata, itheta


def main():
  args = parse_arguments()

  context = zmq.Context()

  # Publisher setup
  publisher_socket = context.socket(zmq.PUB)
  publisher_socket.set_hwm(args.publisher_hwm)
  publisher_socket.bind(args.bind_address_publisher)


  # 1. If data is being simulated, setup simuliation data
  if args.simulate_daq:
    pass

  # 2. Synchronize/handshake with remote
  if args.synchronize_subscribers:
    synchronize_subs(context, args.subscriber_count, args.bind_address_rep)


  #dims=(1920, 1200) 
  dims=(240, 1200) 
  rotation_step=0.25
  builder = flatbuffers.Builder(0)

  print("Sending projections")
  time0 = time.time()
  image = np.random.randint(2, size=dims, dtype=np.uint16)
  total_numb_projs = 1440
  for uniqueId in range(total_numb_projs):
    image[0][0] = uniqueId
    image[-1][-1] = total_numb_projs-uniqueId
    builder.Reset()
    serializer = TraceSerializer.ImageSerializer(builder)
    serialized_data = serializer.serialize(image=image, uniqueId=uniqueId, rotation_step=rotation_step) 
    
    #publisher_socket.send("Projection id={}".format(projection_id).encode())
    publisher_socket.send(serialized_data)

  publisher_socket.send("end_data".encode())
  time1 = time.time()
  print("Done sending; Total time={}".format(time1-time0))


if __name__ == '__main__':
    main()
