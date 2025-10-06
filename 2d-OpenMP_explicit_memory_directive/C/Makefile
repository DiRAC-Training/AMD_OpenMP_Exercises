all: target_data_structured target_data_unstructured target_data_update

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}

LDFLAGS = ${OPENMP_FLAGS} -lm

target_data_structured: target_data_structured.o
	$(CXX) $(LDFLAGS) $^ -o $@

target_data_unstructured: target_data_unstructured.o
	$(CXX) $(LDFLAGS) $^ -o $@

target_data_update: target_data_update.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f target_data_structured target_data_unstructured target_data_update *.o
	rm -rf build