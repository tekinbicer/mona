import zmq


# Arguments
parser = ArgumentParser()
parser.add_argument("-rba", "--router_addr", dest="router_addr", 
  required=True, default="tcp://*:50001",
  help="Router (frontend for REQ)  bind address.")
parser.add_argument("-dba", "--dealer_addr", dest="dealer_addr",
  required=True, default="tcp://*:50002",
  help="Dealer (backend for REP) bind address.")

args = parser.parse_args()


def main():
  context = zmq.Context()

  # Socket facing clients
  frontend = context.socket(zmq.ROUTER)
  frontend.bind(args.router_addr)

  # Socket facing services
  backend  = context.socket(zmq.DEALER)
  backend.bind(args.router_address)

  zmq.proxy(frontend, backend)

  # We never get here…
  frontend.close()
  backend.close()
  context.term()

if __name__ == "__main__":
  main()
