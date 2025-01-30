#pragma once
#include <cuda_runtime.h>

struct SceneConfig {
  float3 space_size;
  float dt;
  float3 G;
  float sphSpacing;
  float initSpacing;
  float smoothing_radius;
  float cell_length;
  int3 cell_size;
  float density;
};
