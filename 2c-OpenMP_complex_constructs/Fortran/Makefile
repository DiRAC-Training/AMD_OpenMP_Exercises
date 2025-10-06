all: saxpy_gpu_paralleldo saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_teams_distribute \
     saxpy_gpu_target_parallel_do saxpy_gpu_split_level saxpy_gpu_collapse 

FC = /opt/rocm-6.3.0/bin/amdflang

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp
OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FFLAGS = -g -O3 ${FREE_FORM_FLAG}

saxpy_gpu_paralleldo.o: saxpy_gpu_paralleldo.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_paralleldo: saxpy_gpu_paralleldo.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target.o: saxpy_gpu_target.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_target: saxpy_gpu_target.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_teams.o: saxpy_gpu_target_teams.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_target_teams: saxpy_gpu_target_teams.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_teams_distribute.o: saxpy_gpu_target_teams_distribute.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_target_teams_distribute: saxpy_gpu_target_teams_distribute.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_parallel_do.o: saxpy_gpu_target_parallel_do.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_target_parallel_do: saxpy_gpu_target_parallel_do.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_split_level.o: saxpy_gpu_split_level.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_split_level: saxpy_gpu_split_level.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_collapse.o: saxpy_gpu_collapse.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o *.mod
	rm -f saxpy_gpu_paralleldo saxpy_gpu_target saxpy_gpu_target_teams
	rm -f saxpy_gpu_target_teams_distribute saxpy_gpu_target_parallel_do
	rm -f saxpy_gpu_split_level saxpy_gpu_collapse
