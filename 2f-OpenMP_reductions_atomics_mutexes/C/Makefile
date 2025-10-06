all: reduction_scalar reduction_array atomic

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

reduction_scalar: reduction_scalar.o
	$(CC) $(LDFLAGS) $^ -o $@

reduction_array: reduction_array.o
	$(CC) $(LDFLAGS) $^ -o $@

atomic: atomic.o
	$(CC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o reduction_scalar reduction_array atomic
