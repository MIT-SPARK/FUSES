#######################
# EXAMPLE EXECUTABLES #
#######################

# Build the fuses-bonnet-Cityscapes executable 
if (bonnet_FOUND)
	add_executable(fuses-bonnet-Cityscapes fuses-bonnet-Cityscapes.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(fuses-bonnet-Cityscapes PRIVATE ${SPECTRA_INCLUDE_DIR})
	target_include_directories(fuses-bonnet-Cityscapes PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(fuses-bonnet-Cityscapes optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	message(STATUS "Type 'make fuses-bonnet-Cityscapes' to build the example.\n")
else(bonnet_FOUND)
  message(STATUS "Skipping fuses-bonnet-Cityscapes example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the fuses-bonnet-batch executable 
if (bonnet_FOUND)
	add_executable(fuses-bonnet-batch fuses-bonnet-batch.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(fuses-bonnet-batch PRIVATE ${SPECTRA_INCLUDE_DIR})
	target_include_directories(fuses-bonnet-batch PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(fuses-bonnet-batch optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	message(STATUS "Type 'make fuses-bonnet-batch' to build the example.\n")
else(bonnet_FOUND)
  message(STATUS "Skipping fuses-bonnet-batch example, since Bonnet is disabled")
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