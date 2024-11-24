set(HEADER_LIST "${CMAKE_SOURCE_DIR}/include/sonne.hpp")

add_library(sonne SHARED sonne.cpp ${HEADER_LIST})

target_include_directories(sonne PUBLIC ../include)

#Set the output directory for the BigInt library
set_target_properties(sonne PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


target_compile_features(sonne PUBLIC cxx_std_20)
target_compile_options(sonne PUBLIC --coverage)

target_link_libraries(sonne gcov)

add_executable(cryptome cryptome.cpp)
target_link_libraries(cryptome sonne ${ARMADILLO_LIBRARIES})

set_target_properties(cryptome PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})