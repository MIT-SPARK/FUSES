#######################
# EXAMPLE EXECUTABLES #
#######################

# Build the fuses-example_bonnet executable 
if (bonnet_FOUND)
	add_executable(fuses-example-bonnet fuses-example-bonnet.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(fuses-example-bonnet PRIVATE ${SPECTRA_INCLUDE_DIR})
	target_include_directories(fuses-example-bonnet PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(fuses-example-bonnet optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	message(STATUS "Type 'make fuses-example-bonnet' to build the example.\n")
else(bonnet_FOUND)
  message(STATUS "Skipping fuses-example_bonnet example, since Bonnet is disabled")
endif(bonnet_FOUND)

########################
# FRONTEND EXECUTABLES #
########################
set(FRONTEND_TARGET_NAME frontend-scripts)
add_subdirectory(frontend-scripts)
message(STATUS "Type 'make ${FRONTEND_TARGET_NAME}' to build the frontend executables.\n")

#######################
# BACKEND EXECUTABLES #
#######################
set(BACKEND_TARGET_NAME backend-scripts)
add_subdirectory(backend-scripts)
message(STATUS "Type 'make ${BACKEND_TARGET_NAME}' to build the backend executables.\n")

####################
# TEST EXECUTABLES #
####################
set(TEST_TARGET_NAME test-scripts)
add_subdirectory(test-scripts)
message(STATUS "Type 'make ${TEST_TARGET_NAME}' to build the test executables.\n")