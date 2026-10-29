# OpenMP target offload on AMD GPUs (ROCm), Fortran.
# The GPU architecture (for example gfx90a on an MI210) is read from rocminfo.
# To set it by hand:  make ROCM_GPU=gfx90a

FC = amdflang

ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef ROCM_GPU
    ROCM_GPU := $(strip $(shell rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]' | sed -e 's/ *Name: *//'))
  endif
  ifeq ($(ROCM_GPU),)
    $(error Could not detect the GPU architecture with rocminfo. Run rocminfo to see why, or set it by hand, for example: make ROCM_GPU=gfx90a)
  endif
endif

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

FFLAGS  = -g -O3 $(OPENMP_FLAGS)
LDFLAGS = $(OPENMP_FLAGS)

.PHONY: all clean

all: openmp_code

openmp_code.o: openmp_code.F90
	$(FC) -c $(FFLAGS) $< -o $@

openmp_code: openmp_code.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o openmp_code *.mod
	rm -rf build