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
add_executable(pipeline_generator.generator src/pipeline_generator.cpp)
target_link_libraries(pipeline_generator.generator PRIVATE Halide::Generator Halide::Tools)

# Filters
add_halide_library(process FROM pipeline_generator.generator TARGETS host FEATURES cuda)

add_executable(app src/main.cpp)
target_include_directories(app PRIVATE)

target_link_libraries(app ${OpenCV_LIBS} ${CUDA_LIBRARIES} process)

# testing 
enable_testing()
#pipline and timing
add_executable(test_pipline test/pipline_timing.cpp)

target_link_libraries(test_pipline ${OpenCV_LIBS} ${CUDA_LIBRARIES} process)
add_test(timimg test_pipline ../test/data/1.tif ../test/data/2.tif) #FIXME

add_executable(test_read_nuc test/read_nuc.cpp src/nuc.cpp)
target_link_libraries(test_read_nuc ${OpenCV_LIBS})
add_test(NUC test_read_nuc ../test/data/nuc_tables.h5)

# KAYA setup
if (DEFINED ENV{KAYA_VISION_POINT_LIB_PATH} AND
    DEFINED ENV{KAYA_VISION_POINT_INCLUDE_PATH})
  # interface test 
  add_executable(test_kaya test/kaya.cpp src/kaya_interface.cpp)
  target_include_directories(test_kaya PRIVATE $ENV{KAYA_VISION_POINT_INCLUDE_PATH})
  target_link_libraries(test_kaya $ENV{KAYA_VISION_POINT_LIB_PATH}/libKYFGLib.so ${OpenCV_LIBS})
  add_test(KAYA test_kaya)
  # interface example
  add_executable(example_kaya test/example_kaya.cpp src/kaya_interface.cpp)
  target_include_directories(example_kaya PRIVATE $ENV{KAYA_VISION_POINT_INCLUDE_PATH})
  target_link_libraries(example_kaya $ENV{KAYA_VISION_POINT_LIB_PATH}/libKYFGLib.so ${OpenCV_LIBS})

  # blur experiment
  add_executable(blur_exp src/blur_exp.cpp src/kaya_interface.cpp)
  target_include_directories(blur_exp PRIVATE $ENV{KAYA_VISION_POINT_INCLUDE_PATH})
  target_link_libraries(blur_exp $ENV{KAYA_VISION_POINT_LIB_PATH}/libKYFGLib.so ${OpenCV_LIBS} ${CUDA_LIBRARIES} process)
  
endif()


