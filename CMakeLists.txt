cmake_minimum_required(VERSION 3.18)
project(pipline LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
# find CUDA
find_package(CUDA REQUIRED)
# Find Halide
find_package(Halide REQUIRED)


# Generator
add_executable(pipeline_generator.generator pipeline_generator.cpp)
target_link_libraries(pipeline_generator.generator PRIVATE Halide::Generator Halide::Tools)

# Filters
add_halide_library(process FROM pipeline_generator.generator TARGETS host FEATURES cuda)


add_executable(app main.cpp)

target_link_libraries(app ${OpenCV_LIBS} ${CUDA_LIBRARIES} process)


# KAYA setup
if (DEFINED ENV{KAYA_VISION_POINT_LIB_PATH} AND
    DEFINED ENV{KAYA_VISION_POINT_INCLUDE_PATH} )
  add_executable(kaya_test test/kaya_test.cpp)
  target_include_directories(kaya_test PRIVATE $ENV{KAYA_VISION_POINT_INCLUDE_PATH})
  target_link_libraries(kaya_test $ENV{KAYA_VISION_POINT_LIB_PATH}/libKYFGLib.so ${OpenCV_LIBS})
endif()

