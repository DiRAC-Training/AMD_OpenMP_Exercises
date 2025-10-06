all: openmp_code_c openmp_code_cxx

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS_CC = -fopenmp --offload-arch=${ROCM_GPU}
OPENMP_FLAGS_CXX = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS_CXX}
CCFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS_CC}
LDFLAGS_CXX = ${OPENMP_FLAGS_CXX} -fno-lto -lm
LDFLAGS_CC = ${OPENMP_FLAGS_CC} -fno-lto -lm

openmp_code_cxx: openmp_code_cxx.o
	${CXX} $(LDFLAGS_CXX) $^ -o $@

openmp_code_c: openmp_code_c.o
	${CC} $(LDFLAGS_CC) $^ -o $@

# Cleanup
clean:
	rm -f *.o openmp_code_c openmp_code_cxx
	rm -rf build
