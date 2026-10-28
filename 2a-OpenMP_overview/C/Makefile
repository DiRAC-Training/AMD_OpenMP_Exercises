# OpenMP target offload on AMD GPUs (ROCm).
# The GPU architecture (for example gfx90a on an MI210) is read from rocminfo.
# To set it by hand:  make ROCM_GPU=gfx90a

CC  = amdclang
CXX = amdclang++

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Run rocminfo to see why, or set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
endif

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CFLAGS   = -g -O3 -fstrict-aliasing $(OPENMP_FLAGS)
CXXFLAGS = -g -O3 -fstrict-aliasing $(OPENMP_FLAGS)
LDFLAGS  = $(OPENMP_FLAGS) -fno-lto
LDLIBS   = -lm

.PHONY: all clean

all: openmp_code_c openmp_code_cxx

openmp_code_c: openmp_code_c.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

openmp_code_cxx: openmp_code_cxx.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f *.o openmp_code_c openmp_code_cxx
	rm -rf build