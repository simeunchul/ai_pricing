cmake_minimum_required(VERSION 3.18)
project(fastmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
  add_compile_options(/O2 /fp:fast /arch:AVX2)
else()
  add_compile_options(-O3 -ffast-math -march=native)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)

pybind11_add_module(_fastmc
  cpp/bindings.cpp
  cpp/mc_kernel.cpp
)

if (OpenMP_CXX_FOUND)
  target_link_libraries(_fastmc PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_fastmc PRIVATE HAS_OPENMP=1)
endif()

install(TARGETS _fastmc LIBRARY DESTINATION fastmc)
