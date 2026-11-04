CXX = amdclang++

all: memorypool

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
  ifndef UMPIRE_PATH
    $(error UMPIRE_PATH is not set. Set it to the Umpire install directory, for example: export UMPIRE_PATH=/opt/umpire)
  endif
endif

ROCM_PATH ?= /opt/rocm
OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
CXXFLAGS = -std=c++20 -g -O3 -fstrict-aliasing $(OPENMP_FLAGS) -D__HIP_PLATFORM_AMD__ -I$(UMPIRE_PATH)/include
LDFLAGS  = $(OPENMP_FLAGS) -fno-lto
LDLIBS   = -L$(UMPIRE_PATH)/lib64 -L$(UMPIRE_PATH)/lib -lumpire -lcamp -lfmt -L$(ROCM_PATH)/lib -lamdhip64 -lstdc++ -lm

.PHONY: all clean

memorypool: memorypool.o ; $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean: ; rm -f *.o memorypool; rm -rf build
