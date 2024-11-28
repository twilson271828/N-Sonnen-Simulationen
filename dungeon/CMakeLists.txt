
add_executable(cryptome cryptome.cpp)
target_link_libraries(cryptome ${ARMADILLO_LIBRARIES})

set_target_properties(cryptome PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})