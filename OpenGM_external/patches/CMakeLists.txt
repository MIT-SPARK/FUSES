
set(externalLibs_scripts
    MaxFlow/patchMaxFlow-v3.02.sh
)

add_custom_target(externalLibs
                  COMMENT "Download, patch and install external libraries used by openGM"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchMaxFlow-v3.02.sh
                   COMMENT "MaxFlow"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaxFlow"
)   

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchTRWS-v1.3.sh
                   COMMENT "TRWS"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/TRWS"
)
