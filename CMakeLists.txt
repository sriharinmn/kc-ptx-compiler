cmake_minimum_required(VERSION 3.20)
project(kc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
add_compile_options(-Wall -Wextra)

# Only the GPU runner needs CUDA (for cuda.h and libcuda). The compiler itself is plain C++.
find_package(CUDAToolkit REQUIRED)

# The compiler: source text in, PTX text out.
add_library(kclib
  src/lexer.cpp
  src/ast.cpp
  src/parser.cpp
  src/sema.cpp
  src/ir.cpp
  src/irgen.cpp
  src/ptx_emit.cpp
  src/analysis.cpp
  src/verify.cpp
  src/divergence.cpp
  src/opt/inline.cpp
  src/opt/simplify.cpp
  src/opt/cse.cpp
  src/opt/licm.cpp
  src/opt/dce.cpp
  src/opt/unroll.cpp
  src/opt/optimize.cpp
  src/compiler.cpp
)
target_include_directories(kclib PUBLIC src)

# Command-line tool: kc file.kc -o file.ptx
add_executable(kc src/main.cpp)
target_link_libraries(kc PRIVATE kclib)

# Tiny wrapper around the CUDA Driver API: load PTX, copy memory, launch kernels.
add_library(gpurt runtime/gpu.cpp)
target_include_directories(gpurt PUBLIC runtime)
target_link_libraries(gpurt PUBLIC CUDA::cuda_driver)

# End-to-end tests: compile .kc programs, run them on the GPU, compare with the CPU.
add_executable(kc-tests tests/test_main.cpp)
target_link_libraries(kc-tests PRIVATE kclib gpurt ptxc)
target_compile_definitions(kc-tests PRIVATE KC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Benchmark: kc vs nvcc on the same kernels. bench/kernels.cu (CUDA C++) is
# compiled to PTX by nvcc; kc's PTX and nvcc's PTX then go through the same
# ptxas, the PTX Compiler API library (runtime/ptx_compiler.cpp).
set(NVCC_PTX ${CMAKE_BINARY_DIR}/bench_kernels.ptx)
add_custom_command(
  OUTPUT ${NVCC_PTX}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=sm_89 -o ${NVCC_PTX} ${CMAKE_SOURCE_DIR}/bench/kernels.cu
  DEPENDS bench/kernels.cu
  COMMENT "nvcc: bench/kernels.cu -> PTX")
add_library(ptxc runtime/ptx_compiler.cpp runtime/linker.cpp)
target_include_directories(ptxc PUBLIC runtime)
# nvJitLink uses the PTX compiler library itself, so it must come first (static libraries link left to right).
target_link_libraries(ptxc PUBLIC CUDA::nvJitLink_static CUDA::nvptxcompiler_static)
add_executable(kc-bench bench/bench_main.cpp ${NVCC_PTX})
target_link_libraries(kc-bench PRIVATE kclib gpurt ptxc)
target_compile_definitions(kc-bench PRIVATE KC_SOURCE_DIR="${CMAKE_SOURCE_DIR}" NVCC_PTX="${NVCC_PTX}")

# Small tools: link PTX files into a cubin, and explain what is inside a cubin.
add_executable(kc-link tools/kc_link.cpp)
target_link_libraries(kc-link PRIVATE ptxc)
add_executable(cubin-info tools/cubin_info.cpp)
# Random program testing: random kernels must give the same output at -O0 and -O1.
add_executable(kc-fuzz tools/fuzz.cpp)
target_link_libraries(kc-fuzz PRIVATE kclib gpurt)
