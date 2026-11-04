CXX = amdclang++

all: alloc_problem

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
endif

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
CXXFLAGS = -g -O3 -fstrict-aliasing $(OPENMP_FLAGS)
LDFLAGS  = $(OPENMP_FLAGS) -fno-lto
LDLIBS   = -lm

.PHONY: all clean

alloc_problem: alloc_problem.o ; $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean: ; rm -f *.o alloc_problem; rm -rf build
