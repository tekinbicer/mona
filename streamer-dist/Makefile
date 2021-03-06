# Compiler
CC = cc

# Flags
CFLAGS = -std=gnu99 -O3 -Wall -Wextra -Wl,--no-undefined
CFLAGS += -g -fPIC

LIBS = -lzmq -lpython3.6m #-lpython2.7
LIBDIRS = -L/home/bicer/miniconda3/envs/tomo/lib
INCLUDES = -I/home/bicer/miniconda3/envs/tomo/include

LDFLAGS = -shared

# Executable/reconstruction objects
SERVER_OBJS = server.o mock_data_acq.o trace_streamer.o

SWIGFILE = server.i
SWIGOBJ = $(SWIGCFILE:c=o)
SWIGCFILE = $(SWIGFILE:.i=_wrap.c)

CCSWIG = swig
SWIGFLAGS = -python
PYTHONINC = -I/home/bicer/miniconda3/envs/tomo/include/python3.6m
#-I/home/software/anaconda3/5.0.0/lib/python3.6/site-packages/numpy/core/include -I/home/software/anaconda3/5.0.0/include/python3.6m #-I/home/beams/USER2BMB/Apps/BlueSky/include/python3.6m #-I/local/tbicer/anaconda2/include/python2.7


# Executables
PROGS = server
TARGET = _tracemq.so

.PHONY: default clean

all: $(TARGET) #$(PROGS)

server: $(COMMON_OBJS) $(SERVER_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(INCLUDES) $(LIBDIRS) $(LIBS)

_tracemq.so: $(COMMON_OBJS) $(SERVER_OBJS) $(SWIGOBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(LIBDIRS) $(SERVER_OBJS) $(LDFLAGS) $(SWIGOBJ) $(LIBS)

swig_server_setup: 
	$(CCSWIG) $(SWIGFLAGS) $(SWIGFILE)

server_wrap.o: swig_server_setup
	$(CC) $(CFLAGS) -c $(SWIGCFILE) $(PYTHONINC) $(INCLUDES)

server.o: server.c
	$(CC) $(CFLAGS) -c server.c $(INCLUDES)

mock_data_acq.o: mock_data_acq.c mock_data_acq.h
	$(CC) $(CFLAGS) -c mock_data_acq.c $(INCLUDES)

trace_streamer.o: trace_streamer.c trace_streamer.h
	$(CC) $(CFLAGS) -c trace_streamer.c $(INCLUDES)

clean:
	rm -f $(PROGS) $(SWIGOBJ) $(SWIGCFILE) tracemq.py tracemq.pyc *.so *.o *.a *~ *.lst *.tmp .pure *.bak *.log
