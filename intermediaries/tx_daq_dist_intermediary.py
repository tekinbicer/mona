import zmq
import argparse
import concurrent.futures as futures
import time


def txparser():
  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--shared_queue_router_addr", dest="router_addr", 
    required=True,
    help="Shared queue router bind address. This is used for synchronization. E.g. \'tcp://*:50001\'")
  parser.add_argument("--shared_queue_dealer_addr", dest="dealer_addr", 
    required=True,
    help="Shared queue dealer connect/bind address. This is used for synchronization. Connection may need to be dealer to rep. E.g. \'tcp://164.54.113.143:50001\'")
  parser.add_argument("--forwarder_xpub_addr", dest="xpub_addr", 
    required=True,
    help="Forwarder xpub address. This is used for data forwarding. E.g. \'tcp://*:50000\'")
  parser.add_argument("--forwarder_xsub_addr", dest="xsub_addr", 
    required=True,
    help="Forwarder xsub address. This is used for data forwarding to workers. Connection may need to be xsub to pub. E.g. \'tcp://164.54.113.143:50000\'")
  parser.add_argument("--zmq_io_threads", dest="zmq_io_threads", 
    required=False, type=int, default=8,
    help="Number of io threads in zmq context. Rule of thumb is 1 thread per 1Gig and number of io threads should be smaller than number of cores. Default value is 8.")

  return parser.parse_args()


def tracex_shared_queue(params):
  context, router_addr, dealer_addr  = params

  # Socket facing clients
  frontend = context.socket(zmq.ROUTER)
  frontend.bind(router_addr)

  # Socket facing services
  backend  = context.socket(zmq.DEALER)
  backend.connect(dealer_addr)

  zmq.proxy(frontend, backend)

  # We never get here…
  frontend.close()
  backend.close()


def tracex_forwarder(params):
  context, xpub_addr, xsub_addr = params

  # Socket facing producers
  frontend = context.socket(zmq.XPUB)
  frontend.bind(xpub_addr)

  # Socket facing consumers
  backend = context.socket(zmq.XSUB)
  backend.connect(xsub_addr) # mona2=164.54.113.143

  zmq.proxy(frontend, backend)

  # We never get here…
  frontend.close()
  backend.close()



def main():
  args = txparser()

  context = zmq.Context(io_threads=args.zmq_io_threads)

  pool = futures.ThreadPoolExecutor()

  shared_queue_params = (context, args.router_addr, args.dealer_addr) #"tcp://*:50001", "tcp://164.54.113.143:50001")
  forwarder_params = (context, args.xpub_addr, args.xsub_addr) #"tcp://*:50000", "tcp://164.54.113.143:50000")

  txforwarder_future = pool.submit(tracex_forwarder, forwarder_params)
  txshared_queue_future = pool.submit(tracex_shared_queue, shared_queue_params)
  print("Forwarder and shared queue are initiated.")

  futures.wait([txforwarder_future, txshared_queue_future])

  # Above is infinite loop so should not be in here!
  context.term()




if __name__ == "__main__":
  main()
