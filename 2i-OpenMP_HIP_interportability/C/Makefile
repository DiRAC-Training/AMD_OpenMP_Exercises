CXX = amdclang++
HIPCC ?= hipcc

EXECUTABLE = daxpy
OBJECTS = daxpy.o daxpy_hip.o

all: $(EXECUTABLE)

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
endif

ROCM_PATH ?= /opt/rocm
OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
CXXFLAGS    = -g -O2 -fPIC -D__HOST_CODE__ $(OPENMP_FLAGS)
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I$(ROCM_PATH)/include -munsafe-fp-atomics -D__DEVICE_CODE__ --offload-arch=$(ROCM_GPU)
LDFLAGS     = $(OPENMP_FLAGS) -L$(ROCM_PATH)/lib -lamdhip64

.PHONY: all clean

daxpy_hip.o: daxpy.cc ; $(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS) ; $(CXX) $(OBJECTS) $(LDFLAGS) -o $@

clean: ; rm -f $(EXECUTABLE) $(OBJECTS) *.hip