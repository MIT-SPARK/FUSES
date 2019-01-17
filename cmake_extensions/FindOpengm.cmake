# This module finds an installed Opengm package.
#
# It sets the following variables:
#  Opengm_FOUND              - Set to false, or undefined, if lemon isn't found.
#  Opengm_INCLUDE_DIR        - Lemon include directory.
#  Opengm_LIBRARIES          - Lemon library files
FIND_PATH(Opengm_INCLUDE_DIR opengm PATHS /usr/include /usr/local/include ${CMAKE_INCLUDE_PATH} ${CMAKE_PREFIX_PATH}/include $ENV{Opengm_ROOT}/include ENV CPLUS_INCLUDE_PATH)

# handle the QUIETLY and REQUIRED arguments and set Opengm_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Opengm DEFAULT_MSG Opengm_INCLUDE_DIR)

MARK_AS_ADVANCED( Opengm_INCLUDE_DIR )

#--------------------------------------------------------------
# options
#--------------------------------------------------------------
OPTION(WITH_TRWS "Include wrapper for TRWS code" ON)
OPTION(WITH_MAXFLOW "Include wrapper for MAXFLOW code" ON)

#--------------------------------------------------------------
# TRWS
#--------------------------------------------------------------
if(WITH_TRWS)
   message(STATUS "build with external inference algorithm TRWS")
   SET(TRWS_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/OpenGM_external/TRWS-v1.3.src-patched" CACHE STRING "TRWS patched source code directory")
   SET(TRWS_SRC_FILES
        ${TRWS_PATCHEDSRCDIR}/minimize.cpp
        ${TRWS_PATCHEDSRCDIR}/MRFEnergy.cpp
        ${TRWS_PATCHEDSRCDIR}/ordering.cpp
        ${TRWS_PATCHEDSRCDIR}/treeProbabilities.cpp)
   add_definitions(-DWITH_TRWS)
   include_directories(${TRWS_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm TRWS")
endif(WITH_TRWS)

#--------------------------------------------------------------
# MaxFlow
#--------------------------------------------------------------
if(WITH_MAXFLOW)
   message(STATUS "build with external inference algorithm MaxFlow")
   SET(MAXFLOW_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/OpenGM_external/MaxFlow-v3.02.src-patched/" CACHE STRING "MAXFLOW patched source code directory")
   SET(MAXFLOW_SRC_FILES
        ${MAXFLOW_PATCHEDSRCDIR}/graph.cpp
        ${MAXFLOW_PATCHEDSRCDIR}/maxflow.cpp)
   add_definitions(-DWITH_MAXFLOW)
   include_directories(${MAXFLOW_PATCHEDSRCDIR})
message(STATUS "MaxFlow directory: ${MAXFLOW_PATCHEDSRCDIR}")
else()
   message(STATUS "build without external inference algorithm MaxFlow")
endif(WITH_MAXFLOW)
