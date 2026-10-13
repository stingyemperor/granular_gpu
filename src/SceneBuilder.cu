#include "SceneBuilder.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Walls of the simulation domain, named by the axis-aligned plane they lie in.
struct Walls {
  bool front;  // z = 0
  bool back;   // z = max
  bool bottom; // y = 0
  bool top;    // y = max
  bool left;   // x = 0
  bool right;  // x = max
};

// Places boundary particles on a grid twice as fine as the neighbor-search
// cells, slightly inset from the domain edges.
void addBoxWalls(std::vector<float3> &pos, const SceneConfig &config,
                 const Walls &walls) {
  const float3 space = config.space_size;
  const int3 n = 2 * config.cellSize();
  auto add = [&](int x, int y, int z) {
    const float3 p =
        make_float3(x, y, z) / make_float3(n - make_int3(1)) * space;
    pos.push_back(0.99f * p + 0.005f * space);
  };

  if (walls.front || walls.back) {
    for (int i = 0; i < n.x; ++i) {
      for (int j = 0; j < n.y; ++j) {
        if (walls.front)
          add(i, j, 0);
        if (walls.back)
          add(i, j, n.z - 1);
      }
    }
  }
  if (walls.bottom || walls.top) {
    for (int i = 0; i < n.x; ++i) {
      for (int j = 0; j < n.z - 2; ++j) {
        if (walls.bottom)
          add(i, 0, j + 1);
        if (walls.top)
          add(i, n.y - 1, j + 1);
      }
    }
  }
  if (walls.left || walls.right) {
    for (int i = 0; i < n.y - 2; ++i) {
      for (int j = 0; j < n.z - 2; ++j) {
        if (walls.left)
          add(0, i + 1, j + 1);
        if (walls.right)
          add(n.x - 1, i + 1, j + 1);
      }
    }
  }
}

Walls wallsFor(SceneType scene) {
  //              front  back   bottom top    left   right
  switch (scene) {
  case SceneType::Piling:
    return {false, false, true, true, false, false};
  case SceneType::Excavator:
  case SceneType::Funnel:
    return {true, true, true, false, true, true};
  case SceneType::Corner:
    return {true, false, true, false, true, false};
  case SceneType::Box:
  case SceneType::BoxOpening:
    break;
  }
  return {true, true, true, true, true, true};
}

std::vector<float3> readPointFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("unable to open obj file: " + filename);
  }
  std::vector<float3> positions;
  float x, y, z;
  while (file >> x >> y >> z) {
    positions.push_back(make_float3(x, y, z));
  }
  return positions;
}

float3 centroid(const std::vector<float3> &points) {
  float3 c = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &p : points) {
    c += p;
  }
  return c / (float)points.size();
}

// Places the obj particles for the scenes that use one.
void placeObjParticles(SceneType scene, std::vector<float3> &obj) {
  if (scene == SceneType::Excavator) {
    for (auto &p : obj) {
      p += make_float3(2.5f, 0.0f, 0.5f);
    }
  } else if (scene == SceneType::Funnel) {
    // rotate 90 degrees about the x axis through the centroid
    const float3 c = centroid(obj);
    const float angle = 90.0f * M_PI / 180.0f;
    const float cos_theta = cosf(angle);
    const float sin_theta = sinf(angle);
    for (auto &p : obj) {
      const float3 d = p - c;
      const float new_y = d.y * cos_theta - d.z * sin_theta;
      const float new_z = d.y * sin_theta + d.z * cos_theta;
      p = make_float3(d.x, new_y, new_z) + c + make_float3(1.25f, 0.3f, 1.5f);
    }
  }
}

std::vector<float3> makeParticleBlock(const SceneConfig &config) {
  const int3 grid = config.particle_grid;
  const float s = config.init_spacing;
  std::vector<float3> pos;
  pos.reserve((size_t)grid.x * grid.y * grid.z);
  for (int y = 0; y < grid.y; ++y) {
    for (int x = 0; x < grid.x; ++x) {
      for (int z = 0; z < grid.z; ++z) {
        pos.push_back(config.particle_origin +
                      make_float3(s * x, s * y, s * z) +
                      config.particle_translation);
      }
    }
  }
  return pos;
}

// Scatters upsample_count fine particles uniformly in a cube of half-width
// particle_radius around each coarse particle.
std::vector<float3> makeUpsampled(const std::vector<float3> &coarse,
                                  const SceneConfig &config) {
  std::vector<float3> pos;
  pos.reserve(coarse.size() * config.upsample_count);
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-config.particle_radius,
                                                     config.particle_radius);
  for (const auto &particle : coarse) {
    for (int n = 0; n < config.upsample_count; ++n) {
      float3 offset =
          make_float3(distribution(generator), distribution(generator),
                      distribution(generator));
      pos.push_back(particle + offset);
    }
  }
  return pos;
}

} // namespace

BuiltScene buildScene(const SceneConfig &config) {
  BuiltScene scene;

  const auto coarse = makeParticleBlock(config);
  auto granular = std::make_shared<GranularParticles>(coarse);
  auto upsampled =
      std::make_shared<GranularParticles>(makeUpsampled(coarse, config));

  std::vector<float3> boundary;
  addBoxWalls(boundary, config, wallsFor(config.scene));
  // the funnel is positioned relative to the untranslated box
  if (config.scene != SceneType::Funnel) {
    for (auto &p : boundary) {
      p += config.boundary_translation;
    }
  }

  // 1 marks obj particles; they are the ones that animate and get exported
  std::vector<int> is_animated(boundary.size(), 0);
  const bool uses_obj =
      config.scene == SceneType::Excavator || config.scene == SceneType::Funnel;
  if (uses_obj && std::filesystem::exists(config.obj_file)) {
    try {
      auto obj = readPointFile(config.obj_file);
      if (!obj.empty()) {
        auto &anim = scene.animation;
        anim.start_index = boundary.size();
        anim.num_particles = obj.size();
        anim.rotation_center = centroid(obj);
        anim.is_animating = config.scene == SceneType::Excavator;

        placeObjParticles(config.scene, obj);
        boundary.insert(boundary.end(), obj.begin(), obj.end());
        is_animated.resize(boundary.size(), 1);
        scene.has_obj = true;
      }
    } catch (const std::exception &e) {
      std::cerr << "Error reading obj particles: " << e.what() << std::endl;
    }
  } else if (uses_obj) {
    std::cerr << "Warning: obj_file \"" << config.obj_file
              << "\" not found; running without it" << std::endl;
  }

  auto boundaries = std::make_shared<GranularParticles>(boundary);
  CUDA_CALL(cudaMemcpy(boundaries->get_is_animated_ptr(), is_animated.data(),
                       is_animated.size() * sizeof(int),
                       cudaMemcpyHostToDevice));

  // only the excavator moves its boundary, which needs a per-step rebuild of
  // the boundary neighbor grid
  const bool moving_boundary = config.scene == SceneType::Excavator;
  scene.system = std::make_unique<GranularSystem>(
      granular, boundaries, upsampled, config.space_size, config.cell_length,
      config.dt, config.gravity, config.cellSize(), config.density,
      config.upsampled_radius, moving_boundary, config.adaptive ? 1 : 0);
  return scene;
}
