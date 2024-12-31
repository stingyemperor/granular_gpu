#include "DArray.hpp"
#include "GranularParticles.hpp"
#include "Particles.hpp"
#include "helper_math.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

__global__ void generate_dots_CUDA(float3 *dot, float3 *pos_color, float3 *pos,
                                   const int num, int *surface) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  dot[i] = pos[i];
  auto w = (powf(1, 2) - 1.0f) * 4.0f;
  w = fminf(w, 1.0f);
  if (surface[i] == 1) {
    pos_color[i] = make_float3(0.34f, 0.46f, 0.7f);
  } else {
    pos_color[i] =
        (1 - w) * make_float3(0.9f) + w * make_float3(1.0f, 0.4f, 0.7f);
  }
}

extern "C" void
generate_dots(float3 *dot, float3 *color,
              const std::shared_ptr<GranularParticles> particles,
              int *surface) {
  generate_dots_CUDA<<<(particles->size() - 1) / block_size + 1, block_size>>>(
      dot, color, particles->get_pos_ptr(), particles->size(), surface);
  cudaDeviceSynchronize();
  CHECK_KERNEL();
  return;
}
