#!/home/beams/USER2BMB/Apps/BlueSky/bin/python

import argparse
import sys
import numpy as np
import zmq
import TraceSerializer
import time
import sys
import os
import pvaccess
import math
sys.path.append(os.path.join(os.path.dirname(__file__), './local'))
import flatbuffers
from dquality.handler_mona import Handler

def parse_arguments():
  parser = argparse.ArgumentParser(
          description='Data Acquisition Process')
  parser.add_argument("--image_pv", default="2bmbPG3:Pva1:Image",
                      help="EPICS PVA image PV name. Default to lyra point grey.")

  parser.add_argument('--bind_address_publisher', default="tcp://*:5560",
                      help='Address to bind publisher.')
  parser.add_argument('--publisher_hwm', type=int, default=0,
                      help='Sets high water mark value for publisher.')
  parser.add_argument('--synchronize_subscribers', action='store_true',
                      help='Waits for all subscribers to join.')
  parser.add_argument('--subscriber_count', type=int,
                      help='Number of expected subscribers.')
  parser.add_argument('--bind_address_rep', default=None,
                      help='Address to bind REP socket (for synchronization)')

  parser.add_argument('--daq_mod', type=int, default=2,
                      help='Data acqusition mod (0=detector; 1=simulate; 2=test)')
  parser.add_argument('--simulation_file', default="../data/hornby_4_x1_conv.h5",
                        help='File name for mock data acquisition. '
                              'Default to shale data.')
  parser.add_argument('--beg_sinogram', type=int, default=0,
                      help='Starting sinogram for reconstruction.')
  parser.add_argument('--num_sinograms', type=int, default=0,
                      help='Number of sinograms to reconstruct.')
  parser.add_argument('--num_sinogram_columns', type=int, default=2048,
                      help='Number of columns per sinogram.')
  parser.add_argument('--num_sinogram_projections', type=int, default=1440,
                      help='Number of projections per sinogram.')
  parser.add_argument('--nprojs_per_subset', default=0, type=int,
                        help='Number of projections per subset. '
                              'Used when simulation file is given, and data acquisition is interlaced.')
  return parser.parse_args()

def synchronize_subs(context, subscriber_count, bind_address_rep):
  print("Synching on: subscriber_count={}; bind_address_rep={}".format(
    subscriber_count, bind_address_rep))
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


def test_daq(publisher_socket, builder,
              rotation_step=0.25, num_sinograms=0, 
              num_sinogram_columns=2048, seq=0,
              num_sinogram_projections=1440):
  print("Sending projections")
  if num_sinograms<1: num_sinograms=2048
  # Randomly generate image data
  dims=(num_sinograms, num_sinogram_columns)
  image = np.array(np.random.randint(2, size=dims), dtype='uint16')

  for uniqueId in range(num_sinogram_projections):
    builder.Reset()
    serializer = TraceSerializer.ImageSerializer(builder)
    serialized_data = serializer.serialize(image=image, uniqueId=uniqueId+7,
                                      itype=serializer.ITypes.Projection, 
                                      rotation_step=rotation_step, seq=seq) 
    seq+=1
    publisher_socket.send(serialized_data)

  return seq



def setup_simulation_data(input_f, beg_sinogram=0, num_sinograms=0, nprojs_per_subset=0):
  import dxchange
  import tomopy as tp
  idata, flat, dark, itheta = dxchange.read_aps_32id(input_f)
  idata = np.array(idata, dtype=np.dtype('uint8'))
  flat = np.array(flat, dtype=np.dtype('uint8'))
  dark = np.array(dark, dtype=np.dtype('uint8'))
  #itheta = tp.sim.project.angles(idata.shape[0])


  if nprojs_per_subset>0:
    #XXX for now return already organized data
    print("loading data")
    idata=np.load("./local.idata.96.npy")
    itheta=np.load("./local.itheta.96.npy")
    dark=np.load("./local.dark.96.npy")
    flat=np.load("./local.flat.96.npy")
    print("done")
    return idata, flat, dark, itheta


    tot_nprojs = idata.shape[0]
    nsubsets= np.int32(math.ceil(1.*tot_nprojs/nprojs_per_subset))
    dlists = []
    tlists = []
    for i in range(nsubsets): 
      dlists.append([])
      tlists.append([])

    print("Number of subsets={}; number of projection per subset={}".format(nsubsets, nprojs_per_subset))

    t0=time.time()
    for i in range(idata.shape[0]):
      dlists[i%nsubsets].extend(idata[i,:,:].flatten().tolist())
      tlists[i%nsubsets].extend(itheta[i].flatten().tolist())
    print("Time to reorganize={}".format(time.time()-t0))
    
    t0=time.time()
    alists=[]
    for i in range(len(dlists)): alists.extend(dlists[i])
    idata = np.array(alists, dtype=np.uint8).reshape(idata.shape)
    alists=[]
    for i in range(len(tlists)): alists.extend(tlists[i])
    itheta = np.array(alists, dtype=np.float32).reshape(itheta.shape)
    print(itheta)
    print("Time to reshape={}".format(time.time()-t0))

    #XXX for now return already organized data
    #print("saving data")
    #np.save("./local.idata.96", idata)
    #np.save("./local.itheta.96", itheta)
    #np.save("./local.dark.96", dark)
    #np.save("./local.flat.96", flat)
    #print("done")


  return idata, flat, dark, itheta

def simulate_daq(publisher_socket, builder, input_f, 
                      beg_sinogram=0, num_sinograms=0, seq=0, slp=0.15, nprojs_per_subset=0):
  # Read image data and theta values
  print("reading data")
  t0=time.time()
  idata, flat, dark, itheta = setup_simulation_data(input_f, beg_sinogram, num_sinograms, nprojs_per_subset)
  print("reading time={}".format(time.time()-t0))

  # Send flat data
  start_index=0
  if flat is not None:
    for uniqueFlatId, flatId in zip(range(start_index, start_index+flat.shape[0]), 
                                    range(flat.shape[0])):
      builder.Reset()
      dflat = flat[flatId]
      print("Publishing flat={}; shape={}".format(uniqueFlatId, flat.shape))
      serializer = TraceSerializer.ImageSerializer(builder)
      itype = serializer.ITypes.WhiteReset if flatId is 0 else serializer.ITypes.White
      serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueFlatId, 
                                        itype=itype,
                                        rotation=0, seq=seq) #, center=10.)
      seq+=1
      publisher_socket.send(serialized_data)
      time.sleep(slp)

  # Send dark data
  start_index+=flat.shape[0]
  if dark is not None:
    for uniqueDarkId, darkId in zip(range(start_index, start_index+dark.shape[0]), 
                                    range(dark.shape[0])):
      builder.Reset()
      dflat = dark[flatId]
      print("Publishing dark={}; shape={}".format(uniqueDarkId, flat.shape))
      serializer = TraceSerializer.ImageSerializer(builder)
      itype = serializer.ITypes.DarkReset if darkId is 0 else serializer.ITypes.Dark
      serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueDarkId, 
                                        itype=itype,
                                        rotation=0, seq=seq) #, center=10.)
      seq+=1
      publisher_socket.send(serialized_data)
      time.sleep(slp)

  # Send projection data
  start_index+=dark.shape[0]
  print(idata.shape, len(itheta), idata.size)
  for uniqueId, projId, rotation in zip(range(start_index, start_index+idata.shape[0]), 
                                        range(idata.shape[0]), itheta):
    builder.Reset()
    proj =  idata[projId]
    print("Publishing={}; shape={}; rotation={}".format(uniqueId, proj.shape, rotation))
    serializer = TraceSerializer.ImageSerializer(builder)
    itype = serializer.ITypes.Projection
    serialized_data = serializer.serialize(image=proj, uniqueId=uniqueId,
                                      itype=itype,
                                      rotation=rotation, seq=seq) #, center=10.)
    seq+=1
    publisher_socket.send(serialized_data)
    time.sleep(slp)

  return seq



class TImageTransfer:
  def __init__(self, publisher_socket, pv_image, builder,
                beg_sinogram=0, num_sinograms=0, seq=0):
    import pvaccess

    self.publisher_socket = publisher_socket
    self.pv_image = pv_image
    self.builder = builder
    self.beg_sinogram = beg_sinogram
    self.num_sinograms = num_sinograms
    self.seq = seq
    self.pv_channel = None
    self.lastImageId=0
    self.mlists=[]


  def __enter__(self):
    self.pv_channel = pvaccess.Channel(self.pv_image)
    x, y = self.pv_channel.get('field()')['dimension']
    self.dims=(y['size'], x['size'])
    labels = [item["name"] for item in self.pv_channel.get('field()')["attribute"]]
    self.theta_key = labels.index("SampleRotary")
    self.scan_delta_key = labels.index("ScanDelta")
    self.start_position_key = labels.index("StartPos")
    self.last_save_dest = labels.index("SaveDest")
    self.imageId = labels.index("ArrayCounter")
    self.flat_count=0
    self.dark_count=0
    self.flat_counter=0
    self.dark_counter=0
    self.sequence=0
    self.tot_lost=0
    print(self.dims)
    if self.num_sinograms>0:
      if (self.beg_sinogram<0) or (self.beg_sinogram+self.num_sinograms>self.dims[0]): 
        raise Exception("Exceeds the sinogram boundary: {} vs. {}".format(
                            self.beg_sinogram+self.num_sinograms, self.dims[0]))
      self.beg_index = self.beg_sinogram*self.dims[1]
      self.end_index = self.beg_sinogram*self.dims[1] + self.num_sinograms*self.dims[1]
    self.pv_channel.subscribe('push_image_data', self.push_image_data)

    # Setup verifier configurations
    self.ver = Handler('/home/beams/USER2BMB/.dquality/dqconfig.ini')
    return self


  def start_monitor(self, smon="value,attribute,uniqueId"):
    self.pv_channel.startMonitor(smon)
    while True: time.sleep(60)  # Forever monitor


  def push_image_data(self, data):
    img = np.frombuffer(data['value'][0]['ubyteValue'], dtype=np.uint8)
    #uniqueId = data['uniqueId']
    uniqueId = data["attribute"][self.imageId]["value"][0]["value"]
    scan_delta = data["attribute"][self.scan_delta_key]["value"][0]["value"]
    start_position = data["attribute"][self.start_position_key]["value"][0]["value"]

    self.builder.Reset()
    serializer = TraceSerializer.ImageSerializer(self.builder)

    itype=data["attribute"][self.last_save_dest]["value"][0]["value"]
    if itype == "/exchange/data":
      print("Projection data: {}".format(uniqueId))
      data_type = 'data'
      itype=serializer.ITypes.Projection
      if self.sequence > 1:
        self.sequence=0
        self.flat_count=self.flat_counter
        self.dark_count=self.dark_counter
        self.flat_counter=0
        self.dark_counter=0
        self.tot_lost=0
    if itype == "/exchange/data_white":
      print("White field: {}".format(uniqueId))
      data_type = 'data_white'
      itype=serializer.ITypes.White
      self.sequence+=1; 
      self.flat_counter+=1
    if itype == "/exchange/data_dark":
      print("Dark field: {}".format(uniqueId))
      data_type = 'data_dark'
      itype=serializer.ITypes.Dark
      self.sequence+=1; 
      self.dark_counter+=1

    #failed = self.ver.handle_data(uniqueId, img, data_type)
    #print ('Verification done: {}'.format(failed))

    # XXX with flat/dark field data uniquId is not a correct value any more
    theta = ((start_position + (uniqueId-(self.flat_count+self.dark_count)))*scan_delta) % 360.0
    diff = (uniqueId-1)-self.lastImageId
    if diff > 0: self.tot_lost += diff
    self.lastImageId = uniqueId
    print("UniqueID={}, Rotation Angle={}; Id Check={}; iType={}; scandelta={}; #flat={}; #dark={}, startposition={}, total_lost={}".format(
          uniqueId, theta, diff, itype, scan_delta, self.flat_count, 
          self.dark_count, start_position, self.tot_lost))
    self.mlists.append([uniqueId, self.flat_count, self.dark_count, 
                        scan_delta, start_position, theta])

    if self.num_sinograms!=0:
      img = img[self.beg_index : self.end_index]
      img = img.reshape((self.num_sinograms, self.dims[1]))
    else: img = img.reshape(self.dims)
    #print("img.shape={}; img={}".format(img.shape, img))

    serialized_data = serializer.serialize(image=img, uniqueId=uniqueId,
                                itype=itype,
                                rotation=theta, seq=self.seq)
    self.publisher_socket.send(serialized_data)
    self.seq+=1


  def __exit__(self, exc_type, exc_value, traceback):
    print("\nTrying to gracefully terminate...")
    self.pv_channel.stopMonitor()
    self.pv_channel.unsubscribe('push_image_data')

    print("Send terminate signal...")
    self.publisher_socket.send("end_data".encode())
    print("Done sending...")
    for i in range(len(self.mlists)):
      print("{}: {}".format(i, self.mlists[i]))
    if exc_type is not None:
      print("{} {} {}".format(exc_type, exc_value, traceback))
      return False 
    return self


    

def main():
  args = parse_arguments()

  # Setup serializer
  builder = flatbuffers.Builder(0)

  # Setup zmq context
  context = zmq.Context()

  # Publisher setup
  publisher_socket = context.socket(zmq.PUB)
  publisher_socket.set_hwm(args.publisher_hwm)
  publisher_socket.setsockopt(zmq.SNDHWM, args.publisher_hwm)
  publisher_socket.bind(args.bind_address_publisher)

  # 1. Synchronize/handshake with remote
  if args.synchronize_subscribers:
    synchronize_subs(context, args.subscriber_count, args.bind_address_rep)

  # 2. Transfer data
  time0 = time.time()
  if args.daq_mod == 0: # Read data from PV
    with TImageTransfer(publisher_socket=publisher_socket,
                        pv_image=args.image_pv, builder=builder, 
                        beg_sinogram=args.beg_sinogram, 
                        num_sinograms=args.num_sinograms, seq=0) as tdet:
      tdet.start_monitor()  # Infinite loop

  elif args.daq_mod == 1: # Simulate data acquisition with a file
    print("simulating on a file")
    simulate_daq(publisher_socket=publisher_socket, 
              input_f=args.simulation_file, builder=builder,
              beg_sinogram=args.beg_sinogram, num_sinograms=args.num_sinograms,
              nprojs_per_subset=args.nprojs_per_subset)
  elif args.daq_mod == 2: # Test data acquisition
    test_daq(publisher_socket=publisher_socket, builder=builder,
              num_sinograms=args.num_sinograms,                       # Y
              num_sinogram_columns=args.num_sinogram_columns,         # X 
              num_sinogram_projections=args.num_sinogram_projections) # Z
  else:
    print("Unknown mode: {}".format(args.daq_mod));

  publisher_socket.send("end_data".encode())
  time1 = time.time()
  print("Done sending; Total time={}".format(time1-time0))


if __name__ == '__main__':
    main()
