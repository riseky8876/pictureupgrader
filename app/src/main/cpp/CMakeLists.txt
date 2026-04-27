# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
set(target pictureupgrader)
project(${target})

# Import OpenCV dependencies
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-4.6.0-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

# Import NCNN dependencies
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-custom-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# NOTE: libncnn.a is compiled with -fvisibility=hidden, which hides typeinfo for
# ncnn::Layer. Any file that subclasses ncnn::Layer with RTTI enabled will fail to
# link with "undefined symbol: typeinfo for ncnn::Layer".
# Fix: compile with -fno-rtti globally (matches how ncnn itself is built),
# and use set_source_files_properties to enforce -fno-rtti on colornet.cpp.
set_target_properties(
    ncnn PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fexceptions"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fexceptions -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

# Define source files
file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

add_library(
        ${target}
        SHARED
        ${srcs} ${hdrs})

include_directories("${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries(
        ${target} ncnn ${OpenCV_LIBS} ${ANDROID_OPENCV_COMPONENTS})
