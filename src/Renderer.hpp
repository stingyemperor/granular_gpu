#pragma once
#include "GranularSystem.hpp"
#include "SceneConfig.hpp"
#include <GL/glew.h>

struct cudaGraphicsResource;

// How the coarse particles are colored (fine particles have mass 0 and are
// always drawn in the sand color).
enum ColorMode {
  COLOR_SURFACE = 0,          // surface particles blue, interior black
  COLOR_SURFACE_DISTANCE = 1, // surface blue, near-surface red, deep black
  COLOR_MASS = 2,             // blue (light) to red (heavy)
};

struct RenderOptions {
  bool show_particles = true;
  bool show_upsampled = true;
  bool show_boundary = true;
  bool show_domain = true;
  int color_mode = COLOR_MASS;
  float point_scale = 1.0f;
  float background[3] = {0.5f, 0.5f, 0.5f};
};

struct Camera {
  float rot_x = 0.0f; // degrees
  float rot_y = 0.0f;
  float zoom = 0.3f; // eye distance is 1 / zoom

  void reset() { *this = Camera(); }
};

class Renderer {
public:
  Renderer() = default;
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // Needs a current GL context.
  void init();
  void release();

  void draw(GranularSystem &system, const SceneConfig &config,
            const Camera &camera, const RenderOptions &options, int width,
            int height);

private:
  // GL position + color buffers shared with CUDA.
  struct PointBuffer {
    GLuint vbo[2] = {0, 0}; // positions, colors
    cudaGraphicsResource *resource[2] = {nullptr, nullptr};
    size_t capacity = 0;

    // Returns true if the buffers were (re)allocated.
    bool ensure(size_t count);
    void map(float3 **pos, float3 **color);
    void unmap();
    void draw(size_t count) const;
    void release();
  };

  void drawParticles(PointBuffer &buffer, const GranularParticles &particles,
                     float max_mass, float min_mass, int color_mode);
  void drawBoundary(const GranularParticles &boundaries);

  PointBuffer _coarse;
  PointBuffer _fine;
  PointBuffer _boundary;
  GLuint _program = 0;
};
