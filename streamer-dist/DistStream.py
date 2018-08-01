#!/home/bicer/.conda/envs/py36/bin/python

import argparse
import numpy as np
import zmq
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './local'))
import flatbuffers
import TraceSerializer

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
  return parser.parse_args()


def synchronize_subs(context, publisher_rep_address):
  sync_socket = context.socket(zmq.REQ)
  sync_socket.connect(publisher_rep_address)

  sync_socket.send(b'') # Send synchronization signal
  sync_socket.recv() # Receive reply
  

def main():
  args = parse_arguments()

  context = zmq.Context()

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
    uniqueId=read_image.UniqueId()
    serializer.info(read_image) # print image information
    seq=read_image.Seq()
    if seq!=total_received: 
      print("Wrong sequence number: {} != {}".format(seq, total_received))
    total_received += 1
    total_size += len(msg)
  time1 = time.time()
    
  # Profile information
  print("Received number of projections: {}".format(total_received))
  print("Rate = {} MB/sec; {} msg/sec".format(
            (total_size/(2**20))/(time1-time0), total_received/(time1-time0)))



if __name__ == '__main__':
  main()
  
