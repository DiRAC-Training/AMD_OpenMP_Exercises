cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKE_Fortran_COMPILER amdflang CACHE STRING "")

project(Memory_pragmas LANGUAGES Fortran)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT ROCM_GPU)
  execute_process(COMMAND rocminfo COMMAND grep -m 1 -E gfx[^0]{1} COMMAND sed -e "s/ *Name: *//"
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROCM_GPU ERROR_QUIET)
endif()
if(NOT ROCM_GPU)
  message(FATAL_ERROR "Could not detect the GPU architecture with rocminfo. Run rocminfo to see why, or set it by hand: cmake -B build -DROCM_GPU=gfx90a")
endif()

string(REPLACE -O2 -O3 CMAKE_Fortran_FLAGS_RELWITHDEBINFO ${CMAKE_Fortran_FLAGS_RELWITHDEBINFO})
set(CMAKE_Fortran_FLAGS_DEBUG "-ggdb")
message(${CMAKE_Fortran_COMPILER_ID})
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp --offload-arch=${ROCM_GPU}")

add_executable(openmp_code openmp_code.F90)