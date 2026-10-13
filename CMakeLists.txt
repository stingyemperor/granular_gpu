cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(granular_gpu VERSION 0.2.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(nlohmann_json 3 REQUIRED)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/imgui/imgui.cpp")
  message(FATAL_ERROR
    "Dear ImGui not found. Run: git submodule update --init --recursive")
endif()

# Dear ImGui with the GLUT + OpenGL2 backends
add_library(imgui STATIC
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glut.cpp
  imgui/backends/imgui_impl_opengl2.cpp
)
target_include_directories(imgui PUBLIC imgui imgui/backends)
target_link_libraries(imgui PUBLIC GLUT::GLUT OpenGL::GL)

add_executable(granular_gpu
  src/main.cu
  src/App.cu
  src/Ui.cu
  src/Renderer.cu
  src/SceneBuilder.cu
  src/Animation.cu
  src/Export.cu
  src/GranularSystem.cu
  src/Solver.cu
  src/SceneConfig.cpp
  src/ShaderUtility.cpp
)

# Lets the program find scenes/, obj/ and shaders from any working directory.
target_compile_definitions(granular_gpu PRIVATE
  GRANULAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

target_compile_options(granular_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall>)

target_link_libraries(granular_gpu PRIVATE
  imgui
  CUDA::cudart
  GLEW::GLEW
  GLUT::GLUT
  OpenGL::GL
  OpenGL::GLU
  nlohmann_json::nlohmann_json
)
