#######################
# BACKEND EXECUTABLES #
#######################
# Build the optimization-FUSES executable
add_executable(optimization-FUSES ./optimization-FUSES.cpp ${FUSES_SOURCES})
target_include_directories(optimization-FUSES PRIVATE ${SPECTRA_INCLUDE_DIR})
target_link_libraries(optimization-FUSES optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(optimization-FUSES PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the optimization-FUSES-DA executable
add_executable(optimization-FUSES-DA ./optimization-FUSES-DA.cpp ${FUSES_SOURCES})
target_include_directories(optimization-FUSES-DA PRIVATE ${SPECTRA_INCLUDE_DIR})
target_link_libraries(optimization-FUSES-DA optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(optimization-FUSES-DA PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the optimization-FUSES2 executable
add_executable(optimization-FUSES2 ./optimization-FUSES2.cpp ${FUSES_SOURCES})
target_include_directories(optimization-FUSES2 PRIVATE ${SPECTRA_INCLUDE_DIR})
target_link_libraries(optimization-FUSES2 optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(optimization-FUSES2 PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the optimization-FUSES2-DA executable
add_executable(optimization-FUSES2-DA ./optimization-FUSES2-DA.cpp ${FUSES_SOURCES})
target_include_directories(optimization-FUSES2-DA PRIVATE ${SPECTRA_INCLUDE_DIR})
target_link_libraries(optimization-FUSES2-DA optimization ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(optimization-FUSES2-DA PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the Expansion-move-alphaExpansion executable
add_executable(expansion-move-alphaExpansion ./expansion-move-alphaExpansion.cpp ${MAXFLOW_SRC_FILES})
target_include_directories(expansion-move-alphaExpansion PUBLIC ${MAXFLOW_PATCHEDSRCDIR}) 
target_link_libraries(expansion-move-alphaExpansion ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(expansion-move-alphaExpansion PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the Belief-propagation-LBP executable
add_executable(belief-propagation-LBP ./belief-propagation-LBP.cpp)
target_link_libraries(belief-propagation-LBP ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(belief-propagation-LBP PROPERTIES EXCLUDE_FROM_ALL 1)

# Build the Belief-propagation-TRWS executable
add_executable(belief-propagation-TRWS ./belief-propagation-TRWS.cpp ${TRWS_SRC_FILES})
target_include_directories(belief-propagation-TRWS PUBLIC ${TRWS_PATCHEDSRCDIR}) 
target_link_libraries(belief-propagation-TRWS ${OpenCV_LIBS} ${GTSAM_LIBRARIES} ${HDF5_LIBRARIES})
set_target_properties(belief-propagation-TRWS PROPERTIES EXCLUDE_FROM_ALL 1)

#################
# Custom target #
#################
add_custom_target(${BACKEND_TARGET_NAME})
add_dependencies(${BACKEND_TARGET_NAME} optimization-FUSES optimization-FUSES-DA optimization-FUSES2 
optimization-FUSES2-DA expansion-move-alphaExpansion belief-propagation-LBP belief-propagation-TRWS)
