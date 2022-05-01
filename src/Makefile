
EXECUTABLE := cudaNeuron

CU_FILES   := neuron.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O0 -g -Wall -std=c++11
#-O3
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O0 -g -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11
#-O3
OBJS=$(OBJDIR)/main.o $(OBJDIR)/neuron.o

PTXS=$(OBJDIR)/neuron.ptx

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS) $(PTXS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(OBJDIR)/%.ptx: %.cu
		$(NVCC) $< -ptx $(NVCCFLAGS) -o $@
