#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
FREE_FORM_FLAG = -ffree-form
FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

compute.o: compute.f90
	$(FC) $(LDFLAGS) -c compute.f90

device_routine.o: device_routine.f90 compute.o
	$(FC) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o compute.o
	$(FC) $(LDFLAGS)  -o device_routine device_routine.o compute.o
# Cleanup
clean:
	rm -f *.o *.mod device_routine
