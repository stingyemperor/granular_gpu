#include "Global.hpp"
#include "ShaderUtility.hpp"
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

__global__ void hello() { printf("Hello World from GPU!\n"); }

// vbo and GL variables
static GLuint particlesVBO;
static GLuint particlesColorVBO;
static GLuint m_particles_program;
static const int m_window_h = 700;
static const int m_fov = 30;
static const float particle_radius = 0.01f;

// view variables
static float rot[2] = {0.0f, 0.0f};
static int mousePos[2] = {-1, -1};
static bool mouse_left_down = false;
static float zoom = 0.3f;

// state variables
static int frameId = 0;
static float totalTime = 0.0f;
bool running = false;

// particle system variables
const float sphSpacing = 0.02f;

void init_system() {
  std::vector<float3> pos;

  for (auto i = 0; i < 36; ++i) {
    for (auto j = 0; j < 24; ++j) {
      for (auto k = 0; k < 24; ++k) {
        auto x = make_float3(0.27f + sphSpacing * j, 0.10f + sphSpacing * i,
                             0.27f + sphSpacing * k);
        pos.push_back(x);
      }
    }
  }
}

int main() {
  std::cout << "Hello World from CPU!" << std::endl;
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}
