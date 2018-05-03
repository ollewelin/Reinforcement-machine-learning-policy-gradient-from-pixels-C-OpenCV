CC = g++
CFLAGS = -g -Wall -O3
SRCS = main.cpp
PROG = exe_main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
