cmake_minimum_required(VERSION 3.20)

project(NeuralNetworkInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo)" FORCE)
endif()

# Find OpenMP for multithreading and timing support
find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenMP_matrix.cpp
)

set(SOURCES
    main.cpp
    matrix/matrix.cpp
    OpenMP_matrix.cpp/buffed_matrix.cpp
    OpenMP_matrix.cpp/omega_buffed_matrix.cpp
    OpenMP_matrix.cpp/nested_buffed_matrix.cpp
    OpenMP_matrix.cpp/omega_matrix.cpp
)

add_executable(nn_inference ${SOURCES})

# Link OpenMP libraries and compiler flags (-fopenmp)
target_link_libraries(nn_inference PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(nn_inference PRIVATE -Wall -Wextra)

add_executable(nn_sweep sweep_bench.cpp)
target_link_libraries(nn_sweep PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(nn_sweep PRIVATE -Wall -Wextra)
