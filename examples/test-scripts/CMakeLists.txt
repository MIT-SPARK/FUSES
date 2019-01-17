####################
# TEST EXECUTABLES #
####################
# Build the fuses-fittingBeta executable
add_executable(fittingBeta ./fuses-fittingBeta.cpp ${FUSES_SOURCES})
target_link_libraries(fittingBeta optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${YAML_CPP_LIBRARIES})
set_target_properties(fittingBeta PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the superpixel-test executable
add_executable(superpixel-test ./superpixel-test.cpp ${FUSES_SOURCES})
target_link_libraries(superpixel-test optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${YAML_CPP_LIBRARIES})
set_target_properties(superpixel-test PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the frontend-bonnet-computeAccuracy executable
add_executable(frontend-bonnet-computeAccuracy ./frontend-bonnet-computeAccuracy.cpp)
target_link_libraries(frontend-bonnet-computeAccuracy optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES})
set_target_properties(frontend-bonnet-computeAccuracy PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the frontend-bonnet-fittingBeta executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-fittingBeta ./frontend-bonnet-fittingBeta.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-fittingBeta PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-fittingBeta optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-fittingBeta PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-fittingBeta example, since Bonnet is disabled")
endif(bonnet_FOUND)

#################
# Custom target #
#################
add_custom_target(${TEST_TARGET_NAME})
add_dependencies(${TEST_TARGET_NAME} fittingBeta superpixel-test frontend-bonnet-computeAccuracy frontend-bonnet-fittingBeta)
