CC=gcc
CXX=g++
CFLAGS = -Wall -O2 -g -Wno-unknown-pragmas
CFLAGS += $(shell aocl compile-config)
#CFLAGS += -I${ALTERAOCLSDKROOT}/board/custom_platform_toolkit/mmd


CXXFLAGS = $(CFLAGS) -std=c++0x
#CXXFLAGS = $(CFLAGS) -std=c++11

#LDFLAGS = $(shell aocl link-config) -lnalla_pcie_mmd
LIBC214 = -L/opt/glibc-2.14/lib
LDFLAGS = $(shell aocl link-config) $(LIBC214)

INSTALL_PATH ?= $$HOME/local

all: bench_pq

bench_pq.o : bench_pq.cpp clwrap.hpp
	$(CXX) -c $^ $(CXXFLAGS) $(LDFLAGS)

bench_pq : bench_pq.cpp clwrap.hpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

#dummy.aocx : dummy.cl
#	aoc -march=emulator -DEMULATOR $<

clean:
	rm -f bench_pq *.o

distclean: clean
	rm -f *~
	rm -rf dummy dummy.aocx dummy.aoco
