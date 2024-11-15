cmake_minimum_required(VERSION 3.16)

project(
    N-Sonnen-Simulationen
    VERSION 1.0
    LANGUAGES CXX)

    
    #Set the build type to Debug if not already set
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug")
    endif()

    enable_testing()

    add_subdirectory(src)
    add_subdirectory(include)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)
    include(${CMAKE_MODULE_PATH}/CodeCoverage.cmake)

    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()

    # Find gcovr
    find_program(GCOVR gcovr)
    if(NOT GCOVR)
        message(FATAL_ERROR "gcovr not found! Please install it.")
    endif()
    