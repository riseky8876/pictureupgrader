cmake_minimum_required(VERSION 3.22.1)

set(target pictureupgrader)
project(${target})

# Import OpenCV dependencies
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-4.6.0-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

# Import NCNN dependencies
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-custom-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# ncnn forces -frtti/-fexceptions via INTERFACE_COMPILE_OPTIONS — override to plain -fexceptions.
# We need -frtti for OpenCV (flann/any.h uses typeid), but colornet.cpp must be compiled
# with -fno-rtti to avoid "undefined symbol: typeinfo for ncnn::Layer" (libncnn.a hides it).
set_target_properties(
    ncnn PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fexceptions"
)

# Global flags: keep -frtti for OpenCV compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -frtti -fopenmp -Os -ffunction-sections -fdata-sections -fvisibility=hidden -w -Wl,--gc-sections")

if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

add_library(${target} SHARED ${srcs} ${hdrs})

# colornet.cpp subclasses ncnn::Layer. libncnn.a hides typeinfo for ncnn::Layer
# (-fvisibility=hidden), so linking with -frtti produces "undefined symbol: typeinfo
# for ncnn::Layer". Compile only this file with -fno-rtti to suppress typeinfo generation.
set_source_files_properties(colornet.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

include_directories("${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries(${target} ncnn ${OpenCV_LIBS} ${ANDROID_OPENCV_COMPONENTS})
