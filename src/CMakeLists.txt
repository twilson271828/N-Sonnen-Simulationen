file(GLOB HEADER_LIST "${CMAKE_SOURCE_DIR}/include/*.hpp")
file(GLOB SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/*.cpp")
#message(STATUS "The value of HEADER_LIST is: ${HEADER_LIST}")

add_library(sonne SHARED ${SOURCE_LIST} ${HEADER_LIST})


set_target_properties(sonne PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories(sonne PUBLIC ../include)


target_compile_features(sonne PUBLIC cxx_std_20)
target_compile_options(sonne PUBLIC --coverage)

target_link_libraries(sonne gcov armadillo)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/src"
  PREFIX "Source Files"
  FILES ${SOURCE_LIST})
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})