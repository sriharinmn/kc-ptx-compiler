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
target_link_libraries(kc-tests PRIVATE kclib gpurt)
target_compile_definitions(kc-tests PRIVATE KC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
