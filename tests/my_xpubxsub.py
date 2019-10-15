#!/usr/bin/python
# -*- coding: UTF-8 -*-

import zmq

def main():

  context = zmq.Context()

  # Socket facing producers
  frontend = context.socket(zmq.XPUB)
  frontend.bind("tcp://*:5559")

  # Socket facing consumers
  backend = context.socket(zmq.XSUB)
  backend.connect("tcp://164.54.113.143:5560") # mona2=164.54.113.143

  zmq.proxy(frontend, backend)

  # We never get here…
  frontend.close()
  backend.close()
  context.term()

if __name__ == "__main__":
  main()
