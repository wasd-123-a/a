cmake_minimum_required(VERSION 3.12)

project(disparity LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV - try system installation first
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Check if ximgproc module is available
if(NOT OpenCV_ximgproc_FOUND)
    message(WARNING "OpenCV ximgproc module not found. Trying alternative approach...")
    # Try to find ximgproc in common locations
    find_path(OPENCV_XIMGPROC_INCLUDE_DIR opencv2/ximgproc/disparity_filter.hpp
        PATHS 
        ${OpenCV_INCLUDE_DIRS}
        /usr/include/opencv4
        /usr/local/include/opencv4
        /usr/include/opencv4/opencv2
        /usr/local/include/opencv4/opencv2
        /usr/include/opencv4.2
        /usr/local/include/opencv4.2
        NO_DEFAULT_PATH
    )
    
endif()

add_executable(disparity ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_executable(edge_detection ${CMAKE_SOURCE_DIR}/src/edge_detection.cpp)
add_executable(bilateral_filter ${CMAKE_SOURCE_DIR}/src/bilateral_filter.cpp)
add_executable(only_wls ${CMAKE_SOURCE_DIR}/src/only_wls.cpp)
add_executable(single_wls ${CMAKE_SOURCE_DIR}/src/single_wls.cpp)

set(EXECUTABLES disparity edge_detection bilateral_filter only_wls single_wls)
foreach(EXE IN LISTS EXECUTABLES)
    target_link_libraries(${EXE} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${EXE} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()

