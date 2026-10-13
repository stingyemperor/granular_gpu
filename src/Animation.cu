#include "Animation.hpp"
#include <vector>

namespace {

__global__ void animateParticles(float3 *positions, const int *is_animated,
                                 int num_particles, float3 translation,
                                 float rotation_angle, float3 rotation_center) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles || is_animated[idx] != 1)
    return;

  float3 pos = positions[idx] + translation;

  // rotate about the z axis through rotation_center
  if (rotation_angle != 0.0f) {
    pos.x -= rotation_center.x;
    pos.y -= rotation_center.y;

    const float angle_rad = rotation_angle * M_PI / 180.0f;
    const float cos_theta = cosf(angle_rad);
    const float sin_theta = sinf(angle_rad);

    const float new_x = pos.x * cos_theta - pos.y * sin_theta;
    const float new_y = pos.x * sin_theta + pos.y * cos_theta;

    pos.x = new_x + rotation_center.x;
    pos.y = new_y + rotation_center.y;
  }

  positions[idx] = pos;
}

float3 centroidOfRange(const GranularParticles &particles, int start,
                       int count) {
  std::vector<float3> positions(count);
  CUDA_CALL(cudaMemcpy(positions.data(), particles.get_pos_ptr() + start,
                       count * sizeof(float3), cudaMemcpyDeviceToHost));
  float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &p : positions) {
    centroid += p;
  }
  return centroid / (float)count;
}

} // namespace

void updateAnimation(AnimationState &state, GranularParticles &boundaries,
                     float dt) {
  if (!state.is_animating)
    return;

  state.animation_time += dt;

  float3 translation = make_float3(0.0f, 0.0f, 0.0f);
  float rotation = 0.0f;

  if (!state.translation_complete) {
    if (state.animation_time <= state.translation_duration) {
      translation = state.translation_direction * state.translation_speed * dt;
    } else {
      state.translation_complete = true;
      state.animation_time = 0.0f;
      if (state.num_particles > 0) {
        state.rotation_center =
            centroidOfRange(boundaries, state.start_index, state.num_particles);
      }
    }
  }

  if (state.translation_complete && state.rotation_angle < state.max_rotation) {
    const float rotation_delta = -state.rotation_speed * dt;
    state.rotation_angle += fabsf(rotation_delta);
    rotation = rotation_delta;

    if (state.rotation_angle >= state.max_rotation) {
      state.is_animating = false;
    }
  }

  if (state.is_animating) {
    const int num = boundaries.size();
    animateParticles<<<(num + block_size - 1) / block_size, block_size>>>(
        boundaries.get_pos_ptr(), boundaries.get_is_animated_ptr(), num,
        translation, rotation, state.rotation_center);
    CUDA_CALL(cudaDeviceSynchronize());
  }
}
