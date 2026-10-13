#pragma once
#include "Animation.hpp"
#include "GranularSystem.hpp"
#include "SceneConfig.hpp"
#include <memory>

struct BuiltScene {
  std::unique_ptr<GranularSystem> system;
  AnimationState animation; // only active for scenes with a moving obj
  bool has_obj = false;     // boundary contains particles from obj_file
};

BuiltScene buildScene(const SceneConfig &config);
