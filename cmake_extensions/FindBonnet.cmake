# This module finds bonnet and all dependencies.
#
# It sets the following variables:
#  bonnet_FOUND              - Set to false, or undefined, if lemon isn't found.
#  bonnet_SOURCES            - bonnet source directory.
#  bonnet_INCLUDES           - bonnet include directory.
#  bonnet_LIBRARIES          - bonnet library files

set(bonnet_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bonnet)

## Find libraries
# find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CUDA_LIBRARIES})

# find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
message(STATUS "NVINFER: ${NVINFER}")
message(STATUS "NVPARSERS: ${NVPARSERS}")
# TODO
set(TensorRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
message(STATUS "TensorRT Headers: ${TensorRT_INCLUDE_DIR}")
include_directories(${TensorRT_INCLUDE_DIR})
set(LIBS ${LIBS} nvinfer nvparsers)
if(NVINFER AND NVPARSERS)
  set(TRT_AVAIL ON)
  add_definitions(-DTRT_AVAIL)
endif()
message("TRT_AVAIL ${TRT_AVAIL}\n")

# find boost
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
message(STATUS "Boost Libs: ${Boost_LIBRARIES}")
message(STATUS "Boost Headers: ${Boost_INCLUDE_DIRS}\n")
include_directories(${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# find opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV Headers: ${OpenCV_INCLUDE_DIRS}\n")
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

# libyaml-cpp
find_package (yaml-cpp REQUIRED)
message(STATUS "YAML Libs: ${YAML_CPP_LIBRARIES}")
message(STATUS "YAML Headers: ${YAML_CPP_INCLUDE_DIR}\n")
include_directories(${YAML_CPP_INCLUDE_DIR})
set(LIBS ${LIBS} ${YAML_CPP_LIBRARIES})

# handle the QUIETLY and REQUIRED arguments and set bonnet_FOUND to TRUE
set(bonnet_SOURCES ${bonnet_DIR}/src/net.cpp 
                   ${bonnet_DIR}/src/bonnet.cpp 
                   ${bonnet_DIR}/src/netTRT.cpp)
set(bonnet_INCLUDES ${bonnet_DIR}/include
                    ${CUDA_INCLUDE_DIRS} 
                    ${TensorRT_INCLUDE_DIR} 
                    ${Boost_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${YAML_CPP_INCLUDE_DIR})
set(bonnet_LIBRARIES ${LIBS})
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(bonnet DEFAULT_MSG bonnet_SOURCES bonnet_INCLUDES bonnet_LIBRARIES)

MARK_AS_ADVANCED( bonnet_SOURCES )
MARK_AS_ADVANCED( bonnet_INCLUDES )
MARK_AS_ADVANCED( bonnet_LIBRARIES )

