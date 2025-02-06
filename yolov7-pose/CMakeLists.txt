cmake_minimum_required(VERSION 3.20)

project(yolov7_pose)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_BUILD_TYPE Debug)

set(Python3_USE_STATIC_LIBS OFF)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY LAST)

# Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

link_directories(${Python3_LIBRARY_DIRS})

# cuda
find_package(CUDA REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
include_directories(${CUDA_INCLUDE_DIRS})

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIBRARY_DIR /usr/lib/x86_64-linux-gnu)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add library libyolo.so
set(LIBYOLO_PATH /app/yolov7-pose/YoloLayer_TRT_v7.0/build)
link_directories(${LIBYOLO_PATH})
set(LIBYOLO libyolo.so)

target_include_directories(yolov7_pose_module PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov7_pose_module PRIVATE 
    ${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${LIBYOLO}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cuda
)


set_target_properties(yolov7_pose_module PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)


add_definitions(-O2 -pthread)