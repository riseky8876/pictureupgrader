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

# Set key properties
# NOTE: -fvisibility=hidden is intentionally removed — it breaks typeinfo linkage
# for classes that extend ncnn::Layer (e.g. Sig17Slice in colornet.cpp).
# The ncnn static library exports typeinfo with default visibility; our subclasses
# must match that visibility or the linker cannot resolve "typeinfo for ncnn::Layer".
set_target_properties(
    ncnn PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -fopenmp -Os -ffunction-sections -fdata-sections -w -Wl,--gc-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frtti -fopenmp -Os -ffunction-sections -fdata-sections -w -Wl,--gc-sections")
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
