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