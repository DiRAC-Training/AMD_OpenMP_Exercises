EXECUTABLE = ./daxpy
OBJECTS = daxpy.o daxpy_hip.o

all: $(EXECUTABLE) daxpy

.PHONY: all clean

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O2 -fPIC -D__HOST_CODE__ ${OPENMP_FLAGS}
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

HIPCC_FLAGS += -munsafe-fp-atomics -D__DEVICE_CODE__
LDFLAGS = ${OPENMP_FLAGS} -L${ROCM_PATH}/lib -lamdhip64


daxpy_hip.hip: daxpy.cc
	cp $^ $@

daxpy_hip.o: daxpy.cc
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.hip
