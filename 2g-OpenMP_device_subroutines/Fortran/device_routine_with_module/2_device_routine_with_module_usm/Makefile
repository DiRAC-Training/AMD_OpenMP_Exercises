#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
FREE_FORM_FLAG = -ffree-form
FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

computemod.o: computemod.f90
	$(FC) $(LDFLAGS) -c computemod.f90

device_routine.o: device_routine.f90 computemod.o
	$(FC) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o computemod.o
	$(FC) $(LDFLAGS)  -o device_routine device_routine.o computemod.o
# Cleanup
clean:
	rm -f *.o *.mod device_routine
