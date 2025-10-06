all: align_two_kernels_64 align_two_kernels_128 align_two_kernels_256 align_two_kernels_512 align_two_kernels_1024

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CXXFLAGS = -g -O3 -std=c++17 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

align_two_kernels_64.o: align_two_kernels.cc
	$(CXX) -c ${CXXFLAGS} -DBLOCKSIZE=64 $^ -o $@

align_two_kernels_64: align_two_kernels_64.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_two_kernels_128.o: align_two_kernels.cc
	$(CXX) -c ${CXXFLAGS} -DBLOCKSIZE=128 $^ -o $@

align_two_kernels_128: align_two_kernels_128.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_two_kernels_256.o: align_two_kernels.cc
	$(CXX) -c ${CXXFLAGS} -DBLOCKSIZE=256 $^ -o $@

align_two_kernels_256: align_two_kernels_256.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_two_kernels_512.o: align_two_kernels.cc
	$(CXX) -c ${CXXFLAGS} -DBLOCKSIZE=512 $^ -o $@

align_two_kernels_512: align_two_kernels_512.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_two_kernels_1024.o: align_two_kernels.cc
	$(CXX) -c ${CXXFLAGS} -DBLOCKSIZE=1024 $^ -o $@

align_two_kernels_1024: align_two_kernels_1024.o
	$(CXX) $^ $(LDFLAGS) -o $@

# Cleanup
clean:
	rm -f *.o align_two_kernels_64 align_two_kernels_128 align_two_kernels_256 align_two_kernels_512 align_two_kernels_1024
