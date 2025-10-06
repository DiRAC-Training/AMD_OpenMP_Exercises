all: saxpy_gpu_parallelfor saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_teams_distribute \
     saxpy_gpu_target_parallel_for saxpy_gpu_split_level saxpy_gpu_collapse 

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS=-fopenmp
OPENMP_OFFLOAD_FLAGS=-fopenmp --offload-arch=${ROCM_GPU}

CFLAGS = -g -O3 -fstrict-aliasing

LDFLAGS = -lm

# Full combined compute directive
saxpy_gpu_parallelfor.o: saxpy_gpu_parallelfor.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_parallelfor: saxpy_gpu_parallelfor.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Target directive
saxpy_gpu_target.o: saxpy_gpu_target.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target: saxpy_gpu_target.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

#Teams clause
saxpy_gpu_target_teams.o: saxpy_gpu_target_teams.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams: saxpy_gpu_target_teams.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

#Distribute clause
saxpy_gpu_target_teams_distribute.o: saxpy_gpu_target_teams_distribute.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams_distribute: saxpy_gpu_target_teams_distribute.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

#Parallel for without the teams distribute clauses
saxpy_gpu_target_parallel_for.o: saxpy_gpu_target_parallel_for.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_parallel_for: saxpy_gpu_target_parallel_for.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

#Split multi-level directive
saxpy_gpu_split_level.o: saxpy_gpu_split_level.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_split_level: saxpy_gpu_split_level.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

#Collapse clause
saxpy_gpu_collapse.o: saxpy_gpu_collapse.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@


clean:
	rm -f *.o
	rm -f saxpy_gpu_parallelfor saxpy_gpu_target saxpy_gpu_target_teams
	rm -f saxpy_gpu_target_teams_distribute saxpy_gpu_target_parallel_for
	rm -f saxpy_gpu_collapse saxpy_gpu_split_level
