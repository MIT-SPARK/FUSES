cmake_minimum_required(VERSION 3.1)

# PROJECT CONFIGURATION
project (FUSES CXX C)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # We require C++ 14

# Enable faster instruction sets (SIMD/AVX)
add_compile_options(-march=native)

# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")
endif()

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place the FUSES library built by this project")

# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

############# DEPENDENCIES #############
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_extensions/)
### Find opencv
find_package(OpenCV 3.4 REQUIRED)
if (OpenCV_FOUND)
  message(STATUS "FOUND OpenCV!")
else (OpenCV_FOUND)
  message(STATUS "CANNOT FOUND OpenCV!")
endif (OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS "${LIBS}" ${OpenCV_LIBS})

### Include the boost directory
find_package(Boost COMPONENTS filesystem REQUIRED)
message(STATUS "Found Boost!")
include_directories(${Boost_INCLUDE_DIRS})

### libyaml-cpp
find_package (yaml-cpp REQUIRED)
message(STATUS "Found YAML Libs")
include_directories(${YAML_CPP_INCLUDE_DIR})
set(LIBS ${LIBS} ${YAML_CPP_LIBRARIES})

### Find the GTSAM package
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam gtsam_unstable)
set (LIBS "${LIBS}" gtsam gtsam_unstable)

### Find MKL (included in GTSAM, so only used for make install)
find_package(MKL)
message(STATUS "MKL Libs: ${MKL_LIBRARIES}")

### Find Optimization library
set(OPTIMIZATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/optimization/ CACHE PATH "Path to top-level directory of Optimization (the one containing CMakeLists.txt)")
# Run this project's CMakeLists file
add_subdirectory(${OPTIMIZATION_DIR})
include_directories(${OPTIMIZATION_DIR})
set(LIBS ${LIBS} optimization)

### FIND SPECTRA
# Spectra doesn't ship with a .cmake file, but it's header-only, so it suffices to simply determine it's top-level directory
set(SPECTRA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spectra/include CACHE PATH "Path to top-level directory of Spectra template library")
#set(SPECTRA_PRIVATE_INCLUDES ${SPECTRA_INCLUDE_DIR} CACHE INTERNAL "")
include_directories(${SPECTRA_INCLUDE_DIR})
message(STATUS "SPECTRA_INCLUDE_DIR! ${SPECTRA_INCLUDE_DIR}")

### Find the OPENGM 
find_package(Opengm REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)
if(NOT HDF5_INCLUDE_DIRS)
  set(HDF5_INCLUDE_DIRS "/usr/include/hdf5/serial" CACHE STRING "Path to hdf5 include directory" FORCE)
  message(STATUS "Setting HDF5_INCLUDE_DIRS to ${HDF5_INCLUDE_DIRS}, as none was specified")
endif()
message(STATUS "  HDF5:  ${HDF5_INCLUDE_DIRS}")
message(STATUS "  Opengm: ${Opengm_INCLUDE_DIR}")
include_directories(${Opengm_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})

### Find bonnet
find_package(Bonnet)
if(bonnet_FOUND)
  message(STATUS "FOUND bonnet!")
  include_directories(${bonnet_INCLUDES})
  set(LIBS ${LIBS} ${bonnet_LIBRARIES})
  add_definitions(-DUSE_BONNET=1)
else()
  message(STATUS "bonnet NOT FOUND")
endif()

############# FUSES PROJECT #############
set(FUSES_SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/src/Fuses.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/Fuses2.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/StiefelMixedProduct.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/StiefelProduct.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/SegmentationFrontEnd.cpp)
set(FUSES_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${FUSES_INCLUDES})

# export FUSES as a library (exclude bonnet)
add_library(fuses SHARED ${FUSES_SOURCES} ${MAXFLOW_SRC_FILES} ${TRWS_SRC_FILES})
target_include_directories(fuses PUBLIC ${SPECTRA_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR} ${MKL_INCLUDE_DIR}
	${Opengm_INCLUDE_DIR} ${MAXFLOW_PATCHEDSRCDIR} ${TRWS_PATCHEDSRCDIR} ${HDF5_INCLUDE_DIRS} ${FUSES_INCLUDES})
target_link_libraries(fuses optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${MKL_LIBRARIES} ${HDF5_LIBRARIES})
export(TARGETS fuses optimization FILE FUSES.cmake)

# allow make install in usr/local
file(GLOB fuses_headers "${FUSES_INCLUDES}/FUSES/*.h")
install(FILES ${fuses_headers} DESTINATION include/FUSES)
install(TARGETS fuses DESTINATION lib)
configure_file(${CMAKE_MODULE_PATH}/FUSESConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/FUSESConfig.cmake)
install(FILES "${PROJECT_BINARY_DIR}/FUSESConfig.cmake" DESTINATION lib/cmake/FUSES)

############# EXECUTABLES #############
add_subdirectory(examples)

### Include the micro for adding test
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake_extensions/GtsamTesting.cmake)

### Add testing
file(GLOB src_cpp_files ${FUSES_SOURCES} ${MAXFLOW_SRC_FILES} ${TRWS_SRC_FILES})
add_subdirectory(tests)
