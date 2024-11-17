find_package(GTest REQUIRED)

add_executable(SonneUnitTest SonneUnitTest.cpp)

target_link_libraries(SonneUnitTest
 PRIVATE
  GTest::GTest
  BigInt)

  
add_test(NAME SonneUnitTest COMMAND SonneUnitTest)
  
target_compile_options(SonneUnitTest PRIVATE --coverage)
target_link_libraries(SonneUnitTest PRIVATE --coverage)

set_target_properties(SonneUnitTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/testbin")

add_custom_target(coverage
    COMMAND ${CMAKE_COMMAND} -E env GCOV_PREFIX=${CMAKE_BINARY_DIR} GCOV_PREFIX_STRIP=3 gcovr -r ${CMAKE_SOURCE_DIR} --html --html-details -o coverage.html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS SonneUnitTest
)