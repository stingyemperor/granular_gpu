#include "DArray.hpp"
#include "GranularParticles.hpp"
#include "Particles.hpp"
#include "helper_math.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

__global__ void generate_dots_CUDA(float3 *dot, float3 *pos_color, float3 *pos,
                                   const int num, int *surface, float *mass,
                                   const float max_mass, const float min_mass,
                                   int show_surface, float *surface_distance) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  dot[i] = pos[i];
  auto w = (powf(1, 2) - 1.0f) * 4.0f;
  w = fminf(w, 1.0f);

  // if (show_surface) {
  //   if (surface[i] == 1) {
  //     pos_color[i] = make_float3(0.34f, 0.46f, 0.7f);
  //   } else {
  //     pos_color[i] = make_float3(0.0f, 0.0f, 0.0f);
  //   }
  // } else {
  //   float norm_mass = (mass[i] - min_mass) / (max_mass - min_mass);

  //   // Linearly interpolate between two colors based on normalized mass
  //   float3 color_min = make_float3(0.0f, 0.0f, 1.0f); // Blue for min mass
  //   float3 color_max = make_float3(1.0f, 0.0f, 0.0f); // Red for max mass

  //   pos_color[i] = (1.0f - norm_mass) * color_min + norm_mass * color_max;

  //   if (mass[i] < 1) {
  //     pos_color[i] = make_float3(0, 0, 0);
  //   }
  // }
  //

  if (show_surface == 0) {
    if (surface[i] == 1) {
      pos_color[i] = make_float3(0.34f, 0.46f, 0.7f);
    } else {
      pos_color[i] = make_float3(0.0f, 0.0f, 0.0f);
    }
  } else if (show_surface == 1) {
    if (surface_distance[i] == 0.0f) {
      pos_color[i] = make_float3(0.34f, 0.46f, 0.7f);
    } else if (surface_distance[i] >= 100.0f) {
      pos_color[i] = make_float3(0.0f, 0.0f, 0.0f);
    } else {
      pos_color[i] = make_float3(1.0f, 0.0f, 0.0f);
    }
  } else if (show_surface == 2) {

    float norm_mass = (mass[i] - min_mass) / (max_mass - min_mass);

    // Linearly interpolate between two colors based on normalized mass
    float3 color_min = make_float3(0.0f, 0.0f, 1.0f); // Blue for min mass
    float3 color_max = make_float3(1.0f, 0.0f, 0.0f); // Red for max mass

    pos_color[i] = (1.0f - norm_mass) * color_min + norm_mass * color_max;

    if (mass[i] < 1) {
      pos_color[i] = make_float3(0, 0, 0);
    }
  }
}

// NOTE: Use for boundary
// __global__ void generate_dots_CUDA(float3 *dot, float3 *pos_color, float3
// *pos,
//                                    const int num, int *surface, float *mass,
//                                    const float max_mass, const float
//                                    min_mass) {
//   const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//   if (i >= num)
//     return;
//
//   dot[i] = pos[i];
//   auto w = (powf(1, 2) - 1.0f) * 4.0f;
//   w = fminf(w, 1.0f);
//
//   if (surface[i] == 1) {
//     pos_color[i] = make_float3(0.34f, 0.46f, 0.7f);
//   } else {
//     pos_color[i] =
//         (1 - w) * make_float3(0.9f) + w * make_float3(1.0f, 0.4f, 0.7f);
//   }
// }

extern "C" void
generate_dots(float3 *dot, float3 *color,
              const std::shared_ptr<GranularParticles> particles, int *surface,
              const float max_mass, const float min_mass, int show_surface) {
  generate_dots_CUDA<<<(particles->size() - 1) / block_size + 1, block_size>>>(
      dot, color, particles->get_pos_ptr(), particles->size(), surface,
      particles->get_mass_ptr(), max_mass, min_mass, show_surface,
      particles->get_surface_distance_ptr());
  cudaDeviceSynchronize();
  CHECK_KERNEL();
  return;
}
