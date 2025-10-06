all = reduction_scalar reduction_array atomic

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
FREE_FORM_FLAG = -ffree-form
FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

reduction_scalar: reduction_scalar.o
	$(FC) $(LDFLAGS) $^ -o $@

reduction_array: reduction_array.o
	$(FC) $(LDFLAGS) $^ -o $@

atomic: atomic.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o reduction_scalar reduction_array atomic *.mod *.bin
