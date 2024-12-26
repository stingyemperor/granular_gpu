cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  "granular_gpu"
  VERSION "0.1.0"
  LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find OpenGL, GLEW, and GLUT
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# Set the source directory
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

# Recursively find all .cpp and .cu files in the src directory
file(
  GLOB_RECURSE
  SOURCE_FILES
  CONFIGURE_DEPENDS
  "${SOURCE_DIR}/*.cpp"
  "${SOURCE_DIR}/*.hpp"
  "${SOURCE_DIR}/*.cu"
  "${SOURCE_DIR}/*.h")

# Create an executable target from all source files
add_executable(main ${SOURCE_FILES})

# Include directories for CUDA, GLEW, GLUT, and OpenGL
target_include_directories(
  main PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS}
               ${GLUT_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})

# Link against CUDA runtime, GLEW, GLUT, and OpenGL libraries
target_link_libraries(
  main PRIVATE CUDA::cudart ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} OpenGL::GL
               GLU # Add GLU library
)

# Set CUDA-specific compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall ")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --no-cuda-version-check")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
# Add a conditional option to avoid passing it to clang
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -forward-unknown-to-host-compiler")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Debug messages
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "GLEW_INCLUDE_DIRS: ${GLEW_INCLUDE_DIRS}")
message(STATUS "GLUT_INCLUDE_DIRS: ${GLUT_INCLUDE_DIRS}")
message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
