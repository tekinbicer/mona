#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import zmq
import time

context = zmq.Context()

socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5560")

time.sleep(3)

for index in range(3):
  time.sleep(2)
  socket.send_string('NASDA:' + time.strftime('%H:%M:%S'))
  time.sleep(2)
  socket.send_string('NASDAQ:' + time.strftime('%H:%M:%S'))

socket.send_string('NASDAQ:STOP')

sys.exit(0)
