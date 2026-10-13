#include "Renderer.hpp"
#include "ShaderUtility.hpp"
#include "helper_math.h"
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace {

const float kFieldOfView = 45.0f; // vertical, degrees

__global__ void colorParticles(float3 *dot, float3 *color, const float3 *pos,
                               const int num, const int *surface,
                               const float *mass, const float max_mass,
                               const float min_mass, int color_mode,
                               const float *surface_distance) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num)
    return;

  dot[i] = pos[i];

  if (color_mode == COLOR_SURFACE) {
    color[i] = surface[i] == 1 ? make_float3(0.34f, 0.46f, 0.7f)
                               : make_float3(0.0f, 0.0f, 0.0f);
  } else if (color_mode == COLOR_SURFACE_DISTANCE) {
    if (surface_distance[i] == 0.0f) {
      color[i] = make_float3(0.34f, 0.46f, 0.7f);
    } else if (surface_distance[i] >= 100.0f) {
      color[i] = make_float3(0.0f, 0.0f, 0.0f);
    } else {
      color[i] = make_float3(1.0f, 0.0f, 0.0f);
    }
  } else {
    const float norm_mass = (mass[i] - min_mass) / (max_mass - min_mass);
    const float3 color_min = make_float3(0.0f, 0.0f, 1.0f);
    const float3 color_max = make_float3(1.0f, 0.0f, 0.0f);
    color[i] = (1.0f - norm_mass) * color_min + norm_mass * color_max;
  }

  // upsampled particles have no mass: draw them in the sand color
  if (mass[i] == 0.0f) {
    color[i] = make_float3(1.0f, 214.0f / 255.0f, 160.0f / 255.0f);
  }
}

} // namespace

bool Renderer::PointBuffer::ensure(size_t count) {
  if (count <= capacity && vbo[0] != 0) {
    return false;
  }
  release();
  capacity = count;
  glGenBuffers(2, vbo);
  for (int i = 0; i < 2; ++i) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(float3), nullptr,
                 GL_DYNAMIC_DRAW);
    CUDA_CALL(cudaGraphicsGLRegisterBuffer(&resource[i], vbo[i],
                                           cudaGraphicsMapFlagsWriteDiscard));
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return true;
}

void Renderer::PointBuffer::map(float3 **pos, float3 **color) {
  size_t bytes = 0;
  CUDA_CALL(cudaGraphicsMapResources(2, resource));
  CUDA_CALL(
      cudaGraphicsResourceGetMappedPointer((void **)pos, &bytes, resource[0]));
  CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)color, &bytes,
                                                 resource[1]));
}

void Renderer::PointBuffer::unmap() {
  CUDA_CALL(cudaGraphicsUnmapResources(2, resource));
}

void Renderer::PointBuffer::draw(size_t count) const {
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::PointBuffer::release() {
  for (int i = 0; i < 2; ++i) {
    if (resource[i]) {
      CUDA_CALL(cudaGraphicsUnregisterResource(resource[i]));
      resource[i] = nullptr;
    }
  }
  if (vbo[0] != 0) {
    glDeleteBuffers(2, vbo);
    vbo[0] = vbo[1] = 0;
  }
  capacity = 0;
}

void Renderer::init() {
  const std::string dir = GRANULAR_SOURCE_DIR "/src/shaders/";
  _program = ShaderUtility::createProgram(dir + "particles.vert",
                                          dir + "particles.frag");
}

void Renderer::release() {
  _coarse.release();
  _fine.release();
  _boundary.release();
  if (_program != 0) {
    glDeleteProgram(_program);
    _program = 0;
  }
}

void Renderer::drawParticles(PointBuffer &buffer,
                             const GranularParticles &particles, float max_mass,
                             float min_mass, int color_mode) {
  const size_t count = particles.size();
  if (count == 0) {
    return;
  }
  buffer.ensure(count);

  float3 *pos, *color;
  buffer.map(&pos, &color);
  colorParticles<<<(count - 1) / block_size + 1, block_size>>>(
      pos, color, particles.get_pos_ptr(), count, particles.get_surface_ptr(),
      particles.get_mass_ptr(), max_mass, min_mass, color_mode,
      particles.get_surface_distance_ptr());
  CHECK_KERNEL();
  buffer.unmap();

  buffer.draw(count);
}

void Renderer::drawBoundary(const GranularParticles &boundaries) {
  const size_t count = boundaries.size();
  if (count == 0) {
    return;
  }
  const bool fresh = _boundary.ensure(count);

  float3 *pos, *color;
  _boundary.map(&pos, &color);
  CUDA_CALL(cudaMemcpy(pos, boundaries.get_pos_ptr(), count * sizeof(float3),
                       cudaMemcpyDeviceToDevice));
  if (fresh) {
    thrust::fill(thrust::device, color, color + count,
                 make_float3(1.0f, 0.0f, 0.0f));
  }
  _boundary.unmap();

  _boundary.draw(count);
}

void Renderer::draw(GranularSystem &system, const SceneConfig &config,
                    const Camera &camera, const RenderOptions &options,
                    int width, int height) {
  height = height > 0 ? height : 1;
  glViewport(0, 0, width, height);
  glClearColor(options.background[0], options.background[1],
               options.background[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(kFieldOfView, (float)width / (float)height, 0.1, 100.0);

  // The piling scene is shifted sideways; follow it with the camera.
  const float3 target = config.scene == SceneType::Piling
                            ? config.particle_translation
                            : make_float3(0.0f, 0.0f, 0.0f);
  const float3 eye = target + make_float3(0.0f, 0.0f, 1.0f / camera.zoom);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, 0, 1, 0);
  glRotatef(camera.rot_x, 1.0f, 0.0f, 0.0f);
  glRotatef(camera.rot_y, 0.0f, 1.0f, 0.0f);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glEnable(GL_MULTISAMPLE_ARB);

  // simulation space [0, space_size] is drawn shifted by -0.5
  glTranslatef(-0.5f, -0.5f, -0.5f);

  if (options.show_domain) {
    const float3 s = config.space_size;
    glPushMatrix();
    glTranslatef(0.5f * s.x, 0.5f * s.y, 0.5f * s.z);
    glScalef(s.x, s.y, s.z);
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
    glLineWidth(2.0f);
    glutWireCube(1.0);
    glPopMatrix();
  }

  glUseProgram(_program);
  // point sprite size in pixels = radius * pointScale / eye distance
  const float point_scale =
      options.point_scale * height / tanf(0.5f * kFieldOfView * M_PI / 180.0f);
  glUniform1f(glGetUniformLocation(_program, "pointScale"), point_scale);
  glEnable(GL_POINT_SPRITE_ARB);
  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  const GLint radius = glGetUniformLocation(_program, "pointRadius");
  const float max_mass = system.get_max_mass();
  const float min_mass = system.get_min_mass();
  if (options.show_particles) {
    glUniform1f(radius, config.particle_radius);
    drawParticles(_coarse, *system.get_particles(), max_mass, min_mass,
                  options.color_mode);
  }
  glUniform1f(radius, config.upsampled_radius);
  if (options.show_upsampled) {
    drawParticles(_fine, *system.get_upsampled(), max_mass, min_mass,
                  options.color_mode);
  }
  if (options.show_boundary) {
    drawBoundary(*system.get_boundaries());
  }

  glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDisable(GL_POINT_SPRITE_ARB);
  glUseProgram(0);
}
