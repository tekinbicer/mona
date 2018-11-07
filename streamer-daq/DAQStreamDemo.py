import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../common/local'))
import argparse
import numpy as np
import zmq
import time
import sys
import TraceSerializer
import flatbuffers
import h5py as h5
import dxchange
import tomopy as tp

def parse_arguments():
  parser = argparse.ArgumentParser(
          description='Data Acquisition Process Simulator')

  parser.add_argument('--publisher_addr', default="tcp://*:50000",
                      help='Publisher addresss of data source process.')
  parser.add_argument('--publisher_hwm', type=int, default=0,
                      help='Sets high water mark value for publisher.')

  parser.add_argument('--synch_addr', default=None,
                      help='Waits for all subscribers to join.')
  parser.add_argument('--synch_count', type=int, default=1,
                      help='Number of expected subscribers.')

  parser.add_argument('--daq_mod', type=int, default=2,
                      help='Data acqusition mod (0=detector; 1=simulate; 2=test)')

  parser.add_argument('--simulation_file', default="../data/tomo_00001.h5",
                        help='File name for mock data acquisition. '
                              'Default to shale data.')
  parser.add_argument('--d_iteration', type=int, default=1,
                      help='Number of iteration on simulated data.')
  parser.add_argument('--beg_sinogram', type=int, default=0,
                      help='Starting sinogram for reconstruction.')
  parser.add_argument('--num_sinograms', type=int, default=0,
                      help='Number of sinograms to reconstruct.')
  parser.add_argument('--num_sinogram_columns', type=int, default=2048,
                      help='Number of columns per sinogram.')
  parser.add_argument('--num_sinogram_projections', type=int, default=1440,
                      help='Number of projections per sinogram.')

  return parser.parse_args()

def synchronize_subs(context, subscriber_count, bind_address_rep):
  # Prepare synch. sockets
  sync_socket = context.socket(zmq.REP)
  sync_socket.bind(bind_address_rep)
  
  counter = 0
  print("Waiting {} subscriber(s) to synchronize...".format(subscriber_count))
  while counter < subscriber_count:
    msg = sync_socket.recv() # wait for subscriber
    sync_socket.send(b'') # reply ack
    counter += 1
    print("Subscriber joined: {}/{}".format(counter, subscriber_count))


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


def setup_simulation_data(input_f, beg_sinogram=0, num_sinograms=0):
  print("Loading tomography data: {}".format(input_f))
  t0=time.time()
  idata, flat, dark, itheta = dxchange.read_aps_32id(input_f)
  idata = np.array(idata, dtype=np.dtype('uint16'))
  flat = np.array(flat, dtype=np.dtype('uint16'))
  dark = np.array(dark, dtype=np.dtype('uint16'))
  print("Projection dataset IO time={:.2f}; dataset shape={}; size={}; Theta shape={};".format(
                             time.time()-t0 , idata.shape, idata.size, itheta.shape))
  return idata, flat, dark, itheta

def serialize_dataset(builder, idata, flat, dark, itheta, seq=0):
  data = []
  start_index=0
  time_ser=0.
  print("Starting serialization")
  if flat is not None:
    for uniqueFlatId, flatId in zip(range(start_index, 
                             start_index+flat.shape[0]), range(flat.shape[0])):
      t_ser0 =  time.time()
      builder.Reset()
      dflat = flat[flatId]
      serializer = TraceSerializer.ImageSerializer(builder)
      itype = serializer.ITypes.WhiteReset if flatId is 0 else serializer.ITypes.White
      serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueFlatId, 
                                        itype=itype,
                                        rotation=0, seq=seq) #, center=10.)
      data.append(serialized_data)
      time_ser += time.time()-t_ser0
      seq+=1
  start_index+=flat.shape[0]

  # dark data
  if dark is not None:
    for uniqueDarkId, darkId in zip(range(start_index, start_index+dark.shape[0]), 
                                    range(dark.shape[0])):
      t_ser0 =  time.time()
      builder.Reset()
      dflat = dark[flatId]
      serializer = TraceSerializer.ImageSerializer(builder)
      itype = serializer.ITypes.DarkReset if darkId is 0 else serializer.ITypes.Dark
      serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueDarkId, 
                                        itype=itype,
                                        rotation=0, seq=seq) #, center=10.)
      time_ser += time.time()-t_ser0
      seq+=1
      data.append(serialized_data)
  start_index+=dark.shape[0]

  # projection data
  for uniqueId, projId, rotation in zip(range(start_index, start_index+idata.shape[0]), 
                                        range(idata.shape[0]), itheta):
    t_ser0 =  time.time()
    builder.Reset()
    proj =  idata[projId]
    serializer = TraceSerializer.ImageSerializer(builder)
    itype = serializer.ITypes.Projection
    serialized_data = serializer.serialize(image=proj, uniqueId=uniqueId,
                                      itype=itype,
                                      rotation=rotation, seq=seq) #, center=10.)
    time_ser += time.time()-t_ser0
    seq+=1
    data.append(serialized_data)
  print("Serialization time={:.2f}".format(time_ser))
  return np.array(data)
  

def simulate_daq_serialized(publisher_socket, builder, input_f, 
                      beg_sinogram=0, num_sinograms=0, seq=0, slp=0,
                      iteration=1):

  idata, flat, dark, itheta = setup_simulation_data(input_f, beg_sinogram, num_sinograms)
  serialized_data = serialize_dataset(builder, idata, flat, dark, itheta)
  del idata, flat, dark
  #print("data shape={}; bytes={}; type={}; serialized_data_len={}".format(serialized_data.shape, serialized_data.nbytes, type(serialized_data[0]), len(serialized_data[0])))

  tot_transfer_size=0
  start_index=0
  time0 = time.time()
  for it in range(iteration): # Simulate data acquisition
    print("Current iteration over dataset: {}/{}".format(it+1, iteration))
    for dchunk in serialized_data:
      publisher_socket.send(dchunk, copy=False)
      tot_transfer_size+=len(dchunk)
  time1 = time.time()

  elapsed_time = time1-time0
  tot_MiBs = (tot_transfer_size*1.)/2**20
  nproj = iteration*len(serialized_data)
  print("Sent number of projections: {}; Total size (MiB): {:.2f}; Elapsed time (s): {:.2f}".format(nproj, tot_MiBs, elapsed_time))
  print("Rate (MiB/s): {:.2f}; (msg/s): {:.2f}".format(tot_MiBs/elapsed_time, nproj/elapsed_time))

  return seq



def simulate_daq(publisher_socket, builder, input_f, 
                      beg_sinogram=0, num_sinograms=0, seq=0, slp=0,
                      iteration=1):

  start_index=0
  time0 = time.time()
  time_ser = 0.
  for it in range(iteration): # Simulate data acquisition
    # Send flat data
    print("Current iteration over dataset: {}/{}".format(it+1, iteration))
    if flat is not None:
      for uniqueFlatId, flatId in zip(range(start_index, 
                               start_index+flat.shape[0]), range(flat.shape[0])):
        t_ser0 =  time.time()
        builder.Reset()
        dflat = flat[flatId]
        #print("Publishing flat={}; shape={}".format(uniqueFlatId, flat.shape))
        serializer = TraceSerializer.ImageSerializer(builder)
        itype = serializer.ITypes.WhiteReset if flatId is 0 else serializer.ITypes.White
        serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueFlatId, 
                                          itype=itype,
                                          rotation=0, seq=seq) #, center=10.)
        time_ser += time.time()-t_ser0
        seq+=1
        publisher_socket.send(serialized_data, copy=False)
        #time.sleep(slp)
    start_index+=flat.shape[0]

    # Send dark data
    if dark is not None:
      for uniqueDarkId, darkId in zip(range(start_index, start_index+dark.shape[0]), 
                                      range(dark.shape[0])):
        t_ser0 =  time.time()
        builder.Reset()
        dflat = dark[flatId]
        #print("Publishing dark={}; shape={}".format(uniqueDarkId, flat.shape))
        serializer = TraceSerializer.ImageSerializer(builder)
        itype = serializer.ITypes.DarkReset if darkId is 0 else serializer.ITypes.Dark
        serialized_data = serializer.serialize(image=dflat, uniqueId=uniqueDarkId, 
                                          itype=itype,
                                          rotation=0, seq=seq) #, center=10.)
        time_ser += time.time()-t_ser0
        seq+=1
        publisher_socket.send(serialized_data, copy=False)
        #time.sleep(slp)
    start_index+=dark.shape[0]

    # Send projection data
    for uniqueId, projId, rotation in zip(range(start_index, start_index+idata.shape[0]), 
                                          range(idata.shape[0]), itheta):
      t_ser0 =  time.time()
      builder.Reset()
      proj =  idata[projId]
      #print("Publishing projection={}; shape={}".format(uniqueId, proj.shape))
      serializer = TraceSerializer.ImageSerializer(builder)
      itype = serializer.ITypes.Projection
      serialized_data = serializer.serialize(image=proj, uniqueId=uniqueId,
                                        itype=itype,
                                        rotation=rotation, seq=seq) #, center=10.)
      time_ser += time.time()-t_ser0
      seq+=1
      publisher_socket.send(serialized_data, copy=False)
      #time.sleep(slp)
  time1 = time.time()
  print("time={}".format(time1-time0))
  tot_MiBs = (iteration*(idata.nbytes+flat.nbytes+dark.nbytes))/2**20
  print("BW (MiB/s)={}".format(tot_MiBs/(time1-time0)))
  print("Serialization time={}".format(time_ser))

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
    print(self.dims)
    if self.num_sinograms>0:
      if (self.beg_sinogram<0) or (self.beg_sinogram+self.num_sinograms>self.dims[0]): 
        raise Exception("Exceeds the sinogram boundary: {} vs. {}".format(
                            self.beg_sinogram+self.num_sinograms, self.dims[0]))
      self.beg_index = self.beg_sinogram*self.dims[1]
      self.end_index = self.beg_sinogram*self.dims[1] + self.num_sinograms*self.dims[1]
    self.pv_channel.subscribe('push_image_data', self.push_image_data)

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
      itype=serializer.ITypes.Projection
      if self.sequence > 1:
        self.sequence=0
        self.flat_count=self.flat_counter
        self.dark_count=self.dark_counter
        self.flat_counter=0
        self.dark_counter=0
    if itype == "/exchange/data_white":
      print("White field: {}".format(uniqueId))
      itype=serializer.ITypes.White
      self.sequence+=1; 
      self.flat_counter+=1
    if itype == "/exchange/data_dark":
      print("Dark field: {}".format(uniqueId))
      itype=serializer.ITypes.Dark
      self.sequence+=1; 
      self.dark_counter+=1

    # XXX with flat/dark field data uniquId is not a correct value any more
    theta = ((start_position + (uniqueId-(self.flat_count+self.dark_count)))*scan_delta) % 360.0
    diff = self.lastImageId-(uniqueId-1)
    self.lastImageId = uniqueId
    print("UniqueID={}, Rotation Angle={}; Id Check={}; iType={}; scandelta={}; #flat={}; #dark={}, startposition={}".format(
                                    uniqueId, theta, diff, itype, scan_delta, self.flat_count, self.dark_count, start_position))
    self.mlists.append([uniqueId, self.flat_count, self.dark_count, scan_delta, start_position, theta])

    if self.num_sinograms!=0:
      img = img[self.beg_index : self.end_index]
      img = img.reshape((self.num_sinograms, self.dims[1]))
    else: img = img.reshape(self.dims)
    print("img.shape={}; img={}".format(img.shape, img))

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
  context = zmq.Context()#(io_threads=2)

  # Publisher setup
  publisher_socket = context.socket(zmq.PUB)
  publisher_socket.set_hwm(args.publisher_hwm)
  publisher_socket.bind(args.publisher_addr)

  # 1. Synchronize/handshake with remote
  if args.synch_addr is not None:
    synchronize_subs(context, args.synch_count, args.synch_addr)

  # 2. Transfer data
  time0 = time.time()
  if args.daq_mod == 0: # Read data from PV
    with TImageTransfer(publisher_socket=publisher_socket,
                        pv_image=args.image_pv, builder=builder, 
                        beg_sinogram=args.beg_sinogram, 
                        num_sinograms=args.num_sinograms, seq=0) as tdet:
      tdet.start_monitor()  # Infinite loop

  elif args.daq_mod == 1: # Simulate data acquisition with a file
    print("Simulating data acquisition on file: {}; iteration: {}".format(args.simulation_file, args.d_iteration))
    simulate_daq_serialized(publisher_socket=publisher_socket, 
              input_f=args.simulation_file, builder=builder,
              beg_sinogram=args.beg_sinogram, num_sinograms=args.num_sinograms,
              iteration=args.d_iteration)
    #simulate_daq(publisher_socket=publisher_socket, 
    #          input_f=args.simulation_file, builder=builder,
    #          beg_sinogram=args.beg_sinogram, num_sinograms=args.num_sinograms,
    #          iteration=args.d_iteration)
  elif args.daq_mod == 2: # Test data acquisition
    test_daq(publisher_socket=publisher_socket, builder=builder,
              num_sinograms=args.num_sinograms,                       # Y
              num_sinogram_columns=args.num_sinogram_columns,         # X 
              num_sinogram_projections=args.num_sinogram_projections) # Z
  else:
    print("Unknown mode: {}".format(args.daq_mod));

  publisher_socket.send("end_data".encode())
  time1 = time.time()
  print("Total time (s): {:.2f}".format(time1-time0))


if __name__ == '__main__':
    main()
