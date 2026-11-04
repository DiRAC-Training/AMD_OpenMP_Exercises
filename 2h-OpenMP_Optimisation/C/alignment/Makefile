CXX = amdclang++

BLOCKSIZES = 64 128 256 512 1024
BINS = $(addprefix align_two_kernels_,$(BLOCKSIZES))
OBJS = $(addsuffix .o,$(BINS))

all: $(BINS)

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
endif

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
CXXFLAGS = -g -O3 -std=c++17 -fstrict-aliasing $(OPENMP_FLAGS)
LDFLAGS  = $(OPENMP_FLAGS) -fno-lto
LDLIBS   = -lm

.PHONY: all clean

$(OBJS): align_two_kernels_%.o: align_two_kernels.cc ; $(CXX) -c $(CXXFLAGS) -DBLOCKSIZE=$* $< -o $@

$(BINS): %: %.o ; $(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

clean: ; rm -f *.o $(BINS); rm -rf build
