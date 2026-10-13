#pragma once
#include "GranularParticles.hpp"
#include "helper_math.h"

// Scripted motion of the obj boundary particles (the excavator bucket):
// slide along translation_direction, then rotate about the z axis.
struct AnimationState {
  bool is_animating = false;
  float animation_time = 0.0f;
  int start_index = 0; // first obj particle in the boundary array
  int num_particles = 0;
  float3 translation_direction = make_float3(-0.1f, 0.0f, 0.0f);
  float translation_speed = 5.0f;
  float translation_duration = 2.5f; // seconds
  bool translation_complete = false;
  float rotation_angle = 0.0f;
  float rotation_speed = 90.0f; // degrees per second
  float max_rotation = 45.0f;   // degrees
  float3 rotation_center = make_float3(0.0f, 0.0f, 0.0f);
};

void updateAnimation(AnimationState &state, GranularParticles &boundaries,
                     float dt);
