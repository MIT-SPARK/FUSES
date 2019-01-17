########################
# FRONTEND EXECUTABLES #
########################
# Build the frontend-bonnet-hdf5 executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5 ./frontend-bonnet-hdf5.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5 PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5 optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5 PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5 example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-bonnet-hdf5-fitting executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5-fitting ./frontend-bonnet-hdf5-fitting.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5-fitting PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5-fitting optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5-fitting PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5-fitting example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-bonnet-hdf5-scalability executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5-scalability ./frontend-bonnet-hdf5-scalability.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5-scalability PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5-scalability optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5-scalability PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5-scalability example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-bonnet-hdf5-gtBinary executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5-gtBinary ./frontend-bonnet-hdf5-gtBinary.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5-gtBinary PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5-gtBinary optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5-gtBinary PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5-gtBinary example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-bonnet-hdf5-gtSP executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5-gtSP ./frontend-bonnet-hdf5-gtSP.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5-gtSP PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5-gtSP optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5-gtSP PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5-gtSP example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-bonnet-hdf5-test executable
if (bonnet_FOUND)
	add_executable(frontend-bonnet-hdf5-test ./frontend-bonnet-hdf5-test.cpp ${FUSES_SOURCES} ${bonnet_SOURCES})
	target_include_directories(frontend-bonnet-hdf5-test PUBLIC ${bonnet_INCLUDES})
	target_link_libraries(frontend-bonnet-hdf5-test optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES} ${bonnet_LIBRARIES})
	set_target_properties(frontend-bonnet-hdf5-test PROPERTIES EXCLUDE_FROM_ALL 1)
else(bonnet_FOUND)
  message(STATUS "Skipping frontend-bonnet-hdf5-test example, since Bonnet is disabled")
endif(bonnet_FOUND)

# Build the frontend-rewrite-hdf5 executable
add_executable(frontend-rewrite-hdf5 ./frontend-rewrite-hdf5.cpp)
target_link_libraries(frontend-rewrite-hdf5 ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(frontend-rewrite-hdf5 PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the frontend-removeNegative-hdf5 executable
add_executable(frontend-removeNegative-hdf5 ./frontend-removeNegative-hdf5.cpp)
target_link_libraries(frontend-removeNegative-hdf5 ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(frontend-removeNegative-hdf5 PROPERTIES EXCLUDE_FROM_ALL 1)

#################
# Custom target #
#################
add_custom_target(${FRONTEND_TARGET_NAME})
add_dependencies(${FRONTEND_TARGET_NAME} frontend-bonnet-hdf5 frontend-bonnet-hdf5-fitting frontend-bonnet-hdf5-scalability 
frontend-bonnet-hdf5-gtBinary frontend-bonnet-hdf5-gtSP frontend-bonnet-hdf5-test frontend-rewrite-hdf5 frontend-removeNegative-hdf5)
