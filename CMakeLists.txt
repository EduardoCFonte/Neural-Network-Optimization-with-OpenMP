cmake_minimum_required(VERSION 3.20)

project(NeuralNetworkInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build (Debug, Release, RelWithDebInfo)" FORCE)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/matrix)

set(SOURCES
    main.cpp
    matrix/matrix.cpp
)

add_executable(nn_inference ${SOURCES})

target_compile_options(nn_inference PRIVATE -Wall -Wextra)