cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(
  "granular_gpu"
  VERSION "0.1.0"
  LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# Set the source directory
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")


# Recursively find all source files
file(
  GLOB_RECURSE
  SOURCE_FILES
  CONFIGURE_DEPENDS
  "${SOURCE_DIR}/*.cpp"
  "${SOURCE_DIR}/*.hpp"
  "${SOURCE_DIR}/*.cu"
  "${SOURCE_DIR}/*.h"
   "${PROJECT_SOURCE_DIR}/imgui/imgui.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/imgui_demo.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/imgui_draw.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/imgui_tables.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/imgui_widgets.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/backends/imgui_impl_glut.h"
    "${PROJECT_SOURCE_DIR}/imgui/backends/imgui_impl_glut.cpp"
   "${PROJECT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl2.cpp"

)

# Create executable target
add_executable(main ${SOURCE_FILES})


# Include directories
target_include_directories(main PRIVATE
    imgui
    imgui/backends
    /usr/include
)

# Include directories
target_include_directories(
  main PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS}
               ${GLUT_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} )

# Link libraries
target_link_libraries(
  main PRIVATE CUDA::cudart ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} OpenGL::GL GLU
)


# CUDA compilation flags
# Note the addition of --extended-lambda flag
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --extended-lambda")

# Enable C++17 for both host and device code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

# Other CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --no-cuda-version-check")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -forward-unknown-to-host-compiler")

# CUDA standards and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Find Thrust (usually comes with CUDA)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

# Link Thrust to our target
target_link_libraries(main PRIVATE Thrust)


# Debug messages
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "GLEW_INCLUDE_DIRS: ${GLEW_INCLUDE_DIRS}")
message(STATUS "GLUT_INCLUDE_DIRS: ${GLUT_INCLUDE_DIRS}")
message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
