cmake_minimum_required(VERSION 3.22.1)

set(target pictureupgrader)
project(${target})

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-4.6.0-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-custom-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Suppress ncnn's injected compile options so we control rtti ourselves
set_target_properties(ncnn PROPERTIES INTERFACE_COMPILE_OPTIONS "-fexceptions")

# Global: -frtti required by OpenCV (flann/any.h uses typeid)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")

if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif()

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

add_library(${target} SHARED ${srcs} ${hdrs})

# colornet_impl.cpp contains Sig17Slice (subclass of ncnn::Layer).
# libncnn.a is built with -fno-rtti so typeinfo for ncnn::Layer does not exist.
# Compiling colornet_impl.cpp with -fno-rtti prevents typeinfo generation entirely,
# avoiding "undefined symbol: typeinfo for ncnn::Layer".
# colornet.cpp (the public API wrapper) stays -frtti so name mangling matches callers.
set_source_files_properties(colornet_impl.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

include_directories("${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries(${target} ncnn ${OpenCV_LIBS} ${ANDROID_OPENCV_COMPONENTS})
