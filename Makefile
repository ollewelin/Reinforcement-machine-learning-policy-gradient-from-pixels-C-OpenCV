CC = g++
#CFLAGS = -g -Wall -O3 -std=c++11
CFLAGS = -g -Wall -O3
SRCS = main.cpp
PROG = exe_main

#OPENCV = `pkg-config opencv --cflags --libs`
OPENCV = `pkg-config --cflags --libs opencv4`

LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
