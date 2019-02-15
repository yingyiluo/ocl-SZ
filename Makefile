
#PLATFORM = intelfpga
PLATFORM = intelgpu

USE_MEAN = no
#USE_MEAN = yes

ifeq ($(PLATFORM),intelfpga)
CXX=g++
CFLAGS = -Wall -O2 -g -Wno-unknown-pragmas
CFLAGS += $(shell aocl compile-config)
CXXFLAGS = $(CFLAGS) -std=c++11 -DENABLE_INTELFPGA
LDFLAGS = $(shell aocl link-config)
endif

ifeq ($(PLATFORM),intelgpu)
CXX=g++
CXXFLAGS = -Wall -O2 -g -std=gnu++0x -DENABLE_INTELGPU
LDFLAGS = -lOpenCL
endif

ifeq ($(USE_MEAN), yes)
CXXFLAGS += -DUSE_MEAN
endif

ifneq (,$(SDK))
CXXFLAGS += -I$(SDK)/include
LDFLAGS += -L$(SDK)/lib64
endif

#LIBC214 = -L/opt/glibc-2.14/lib
#LDFLAGS = $(shell aocl link-config) $(LIBC214)

INSTALL_PATH ?= $$HOME/local

all: bench_pq

bench_pq.o : bench_pq.cpp clwrap.hpp bench_pq.hpp
	$(CXX) -c $^ $(CXXFLAGS) $(LDFLAGS)

bench_pq : bench_pq.cpp clwrap.hpp bench_pq.hpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

#dummy.aocx : dummy.cl
#	aoc -march=emulator -DEMULATOR $<

clean:
	rm -f bench_pq *.o

distclean: clean
	rm -f *~
	rm -rf dummy dummy.aocx dummy.aoco
