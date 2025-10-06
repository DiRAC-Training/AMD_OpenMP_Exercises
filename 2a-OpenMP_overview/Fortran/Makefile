all:openmp_code

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FFLAGS = -g -O3 -Mfreeform ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

openmp_code.o: openmp_code.F90
	$(FC) -c $(FFLAGS) $^

openmp_code: openmp_code.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o openmp_code *.mod
	rm -rf build
