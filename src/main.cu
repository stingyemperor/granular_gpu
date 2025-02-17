#include "Global.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"
#include "ShaderUtility.hpp"
#include "VBO.hpp"
#include "helper_math.h"
#include <GL/freeglut.h>
#include <cerrno>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// interactivity
bool picking_mode = false;
float3 pick_center = make_float3(0.0f, 0.0f, 0.0f);
float pick_radius = 0.03f; // Radius of picking sphere
DArray<int> picked_particles(1);
int num_picked = 0; // Keep track of number of picked particles
float3 last_pick_pos = make_float3(0.0f, 0.0f, 0.0f);
std::string obj_file;
std::vector<float3> stored_additional_particles;
int is_adaptive = 1;
std::string save_file;

using json = nlohmann::json;
// vbo and GL variables
static GLuint particlesVBO;
static GLuint particlesColorVBO;
static GLuint upsampledParticlesVBO;
static GLuint upsampledParticlesColorVBO;
static const float upsampled_particle_radius = 0.005f; // Half
static GLuint m_particles_program;
static const int m_window_h = 1600;
static const int m_fov = 30;
static const float particle_radius = 0.01f;
float density = 238732.4146f;
static size_t last_particle_count = 0;
int scene = 0;
float3 boundary_translation = make_float3(0.0f, 0.0f, 0.0f);
float3 particle_translation = make_float3(0.0f, 0.0f, 0.0f);

// view variables
static float rot[2] = {0.0f, 0.0f};
static int mousePos[2] = {-1, -1};
static bool mouse_left_down = false;
static float zoom = 0.3f;

// state variables
static int frameId = 0;
static float totalTime = 0.0f;
bool running = false;
int show_surface = 2;

// particle system variables
std::shared_ptr<GranularSystem> p_system;
float3 space_size = make_float3(1.8f, 1.8f, 1.8f);
float3 box_size = make_float3(1.8f, 1.8f, 1.8f);
// const float3 space_size = make_float3(1.5f, 1.5f, 1.5f);
float dt = 0.002f;
float3 G = make_float3(0.0f, -9.8f, 0.0f);
float sphSpacing = 0.02f;
float initSpacing = 0.030f;
float smoothing_radius = 2.0f * sphSpacing;
float cell_length = 1.01f * smoothing_radius;
// const float cell_length = 0.02f;
int3 cell_size = make_int3(ceil(space_size.x / cell_length),
                           ceil(space_size.y / cell_length),
                           ceil(space_size.z / cell_length));

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
  float3 box_size;
  float3 boundary_translation;
  float3 particle_translation;
  int scene;
  std::string obj_file;
  int is_adaptive;
  std::string save_file;
};

struct AnimationState {
  bool is_animating = false;
  float animation_time = 0.0f;
  int start_index = 0;
  int num_particles = 0;
  float3 translation_direction =
      make_float3(-0.1f, 0.0f, 0.0f); // Translation direction
  float translation_speed = 5.0f;
  bool translation_complete = false;
  float rotation_angle = 0.0f;
  float rotation_speed = 90.0f; // degrees per second
  float3 rotation_center = make_float3(0.0f, 0.0f, 0.0f);
};

AnimationState animation_state;
enum Scene { PILING = 0, BOX = 1, EXCAVATOR = 2, FUNNEL = 3, CORNER = 4 };

SceneConfig loadSceneConfig(const std::string &config_file) {
  SceneConfig config;

  std::ifstream file(config_file);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open config file: " + config_file);
  }

  json j;
  file >> j;

  // Read space size
  config.space_size = make_float3(j["space_size"]["x"], j["space_size"]["y"],
                                  j["space_size"]["z"]);

  config.dt = j["dt"];

  // Read gravity
  config.G =
      make_float3(j["gravity"]["x"], j["gravity"]["y"], j["gravity"]["z"]);

  config.box_size =
      make_float3(j["box_size"]["x"], j["box_size"]["y"], j["box_size"]["z"]);

  config.sphSpacing = j["sph_spacing"];
  config.initSpacing = j["init_spacing"];
  config.smoothing_radius = j["smoothing_radius"];
  config.cell_length = j["cell_length"];
  config.density = j["density"];
  config.scene = j["scene"];
  config.boundary_translation = make_float3(j["boundary_translation"]["x"],
                                            j["boundary_translation"]["y"],
                                            j["boundary_translation"]["z"]);
  config.particle_translation = make_float3(j["particle_translation"]["x"],
                                            j["particle_translation"]["y"],
                                            j["particle_translation"]["z"]);

  // Calculate cell size based on space size and cell length
  config.cell_size = make_int3(ceil(config.space_size.x / config.cell_length),
                               ceil(config.space_size.y / config.cell_length),
                               ceil(config.space_size.z / config.cell_length));

  config.obj_file = j["obj_file"];
  config.is_adaptive = j["is_adaptive"];
  config.save_file = j["save_file"];

  return config;
}

std::vector<float3> readBoundaryParticlesFromFile(const std::string &filename,
                                                  int &start_index) {
  std::vector<float3> positions;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Unable to open boundary particles file: " +
                             filename);
  }

  float x, y, z;
  while (file >> x >> y >> z) {
    positions.push_back(make_float3(x, y, z));
  }

  // Store the start index before adding new particles
  start_index = positions.size();

  file.close();
  return positions;
}

void saveBoundaryParticlesToVTK(
    const std::shared_ptr<GranularParticles> &boundary_particles,
    const std::shared_ptr<GranularParticles> &upsampled_particles,
    int frameId) {
  std::string base_dir = save_file + "_boundary";
  std::filesystem::create_directories(base_dir);
  std::string filename = base_dir + "/" + std::to_string(frameId) + ".vtk";

  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Get positions from both boundary and upsampled particles
  std::vector<float3> boundary_positions(boundary_particles->size());
  std::vector<int> is_animated(boundary_particles->size());
  std::vector<float3> upsampled_positions(upsampled_particles->size());

  CUDA_CALL(cudaMemcpy(
      boundary_positions.data(), boundary_particles->get_pos_ptr(),
      boundary_positions.size() * sizeof(float3), cudaMemcpyDeviceToHost));

  CUDA_CALL(
      cudaMemcpy(is_animated.data(), boundary_particles->get_is_animated_ptr(),
                 is_animated.size() * sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaMemcpy(
      upsampled_positions.data(), upsampled_particles->get_pos_ptr(),
      upsampled_positions.size() * sizeof(float3), cudaMemcpyDeviceToHost));

  // Filter animated boundary particles
  std::vector<float3> animated_positions;
  for (size_t i = 0; i < boundary_positions.size(); ++i) {
    if (is_animated[i] == 1) {
      animated_positions.push_back(boundary_positions[i]);
    }
  }

  if (animated_positions.empty()) {
    std::cerr << "Warning: Frame " << frameId
              << " contains no animated boundary particles" << std::endl;
    return;
  }

  // Calculate centroid of upsampled particles (same as in
  // saveUpsampledPositionsToVTK)
  std::vector<float3> valid_upsampled;
  for (const auto &pos : upsampled_positions) {
    if (!isnan(pos.x) && !isnan(pos.y) && !isnan(pos.z)) {
      valid_upsampled.push_back(pos);
    }
  }

  float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &pos : valid_upsampled) {
    centroid += pos;
  }
  centroid = centroid / (float)valid_upsampled.size();

  // Find minimum y value after centering (using upsampled particles)
  float min_y = 1000000.f;
  for (const auto &pos : valid_upsampled) {
    float centered_y = pos.y - centroid.y;
    min_y = fminf(min_y, centered_y);
  }

  // Calculate y offset to ensure all y values are positive
  float y_offset = min_y < 0.0f ? -min_y : 0.0f;

  // Write VTK header
  outFile << "# vtk DataFile Version 3.0\n";
  outFile << "Animated Boundary Particles\n";
  outFile << "ASCII\n";
  outFile << "DATASET UNSTRUCTURED_GRID\n";

  // Write points using the same centering as upsampled particles
  outFile << "POINTS " << animated_positions.size() << " float\n";
  for (const auto &pos : animated_positions) {
    float3 centered_pos = make_float3(
        pos.x - centroid.x, pos.y - centroid.y + y_offset, pos.z - centroid.z);
    outFile << centered_pos.x << " " << centered_pos.z << " " << centered_pos.y
            << "\n";
  }

  // Write cells
  outFile << "CELLS " << animated_positions.size() << " "
          << animated_positions.size() * 2 << "\n";
  for (size_t i = 0; i < animated_positions.size(); i++) {
    outFile << "1 " << i << "\n";
  }

  // Write cell types
  outFile << "CELL_TYPES " << animated_positions.size() << "\n";
  for (size_t i = 0; i < animated_positions.size(); i++) {
    outFile << "1\n";
  }

  outFile.close();
}

void saveAdditionalBoundaryParticlesToVTK(
    const std::shared_ptr<GranularParticles> &boundary_particles,
    const std::vector<float3> &original_additional_particles,
    const std::string &filename) {
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write VTK header
  outFile << "# vtk DataFile Version 3.0\n";
  outFile << "Additional Boundary Particles\n";
  outFile << "ASCII\n";
  outFile << "DATASET UNSTRUCTURED_GRID\n";

  // Write points
  outFile << "POINTS " << original_additional_particles.size() << " float\n";
  for (const auto &pos : original_additional_particles) {
    outFile << pos.x << " " << pos.z << " " << pos.y << "\n";
  }

  // Write cells (each particle is a vertex cell)
  outFile << "CELLS " << original_additional_particles.size() << " "
          << original_additional_particles.size() * 2 << "\n";
  for (size_t i = 0; i < original_additional_particles.size(); i++) {
    outFile << "1 " << i << "\n";
  }

  // Write cell types (VTK_VERTEX = 1)
  outFile << "CELL_TYPES " << original_additional_particles.size() << "\n";
  for (size_t i = 0; i < original_additional_particles.size(); i++) {
    outFile << "1\n";
  }

  outFile.close();
}

void saveUpsampledPositionsToVTK(
    const std::shared_ptr<GranularParticles> &upsampled_particles,
    int frameId) {

  // Create base directory based on adaptive setting
  std::string base_dir = save_file + (is_adaptive ? "_adaptive" : "_normal");

  // Create directories if they don't exist
  std::filesystem::create_directories(base_dir);

  // Create filename with frame number
  std::string filename = base_dir + "/" + std::to_string(frameId) + ".vtk";

  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Get positions from device
  std::vector<float3> positions(upsampled_particles->size());
  CUDA_CALL(cudaMemcpy(positions.data(), upsampled_particles->get_pos_ptr(),
                       positions.size() * sizeof(float3),
                       cudaMemcpyDeviceToHost));

  // Check for NaN values and count valid particles
  std::vector<float3> valid_positions;
  valid_positions.reserve(positions.size());

  for (const auto &pos : positions) {
    if (!isnan(pos.x) && !isnan(pos.y) && !isnan(pos.z)) {
      valid_positions.push_back(pos);
    }
  }

  if (valid_positions.empty()) {
    std::cerr << "Warning: Frame " << frameId
              << " contains no valid positions (all NaN)" << std::endl;
    return;
  }

  if (valid_positions.size() < positions.size()) {
    std::cerr << "Warning: Frame " << frameId << " - Filtered out "
              << (positions.size() - valid_positions.size())
              << " particles with NaN values" << std::endl;
  }

  // Calculate centroid using only valid positions
  float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &pos : valid_positions) {
    centroid += pos;
  }
  centroid = centroid / (float)valid_positions.size();

  // Find minimum y value after centering
  float min_y = 1000000.f;
  for (const auto &pos : valid_positions) {
    float centered_y = pos.y - centroid.y;
    min_y = fminf(min_y, centered_y);
  }

  // Calculate y offset to ensure all y values are positive
  float y_offset = min_y < 0.0f ? -min_y : 0.0f;

  // Write VTK header
  outFile << "# vtk DataFile Version 3.0\n";
  outFile << "Upsampled Granular Particles\n";
  outFile << "ASCII\n";
  outFile << "DATASET UNSTRUCTURED_GRID\n";

  // Write points (centered around origin with positive y values)
  outFile << "POINTS " << valid_positions.size() << " float\n";
  for (const auto &pos : valid_positions) {
    float3 centered_pos = make_float3(
        pos.x - centroid.x, pos.y - centroid.y + y_offset, pos.z - centroid.z);
    outFile << centered_pos.x << " " << centered_pos.z << " " << centered_pos.y
            << "\n";
  }

  // Write cells
  outFile << "CELLS " << valid_positions.size() << " "
          << valid_positions.size() * 2 << "\n";
  for (size_t i = 0; i < valid_positions.size(); i++) {
    outFile << "1 " << i << "\n";
  }

  // Write cell types
  outFile << "CELL_TYPES " << valid_positions.size() << "\n";
  for (size_t i = 0; i < valid_positions.size(); i++) {
    outFile << "1\n";
  }

  outFile.close();
}
void saveFrameTimes(const std::vector<float> &frame_times) {
  std::ofstream outFile("frame_times_adaptive.txt");
  for (float time : frame_times) {
    outFile << time << "\n";
  }
  outFile.close();
}

void saveUpsampledPositionsToCSV(
    const std::shared_ptr<GranularParticles> &upsampled_particles,
    int frameId) {
  // Create filename with frame number
  std::string filename = "data/" + std::to_string(frameId) + ".csv";
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Get positions from device
  std::vector<float3> positions(upsampled_particles->size());
  CUDA_CALL(cudaMemcpy(positions.data(), upsampled_particles->get_pos_ptr(),
                       positions.size() * sizeof(float3),
                       cudaMemcpyDeviceToHost));

  // Write positions to CSV
  for (const auto &pos : positions) {
    outFile << pos.x << "," << pos.y << "," << pos.z << "\n";
  }

  outFile.close();
}

void init_granular_system() {
  // NOTE: Fill up the initial positions of the particles
  std::vector<float3> pos;
  // 36 24 24
  for (auto i = 0; i < box_size.x; ++i) {
    for (auto j = 0; j < box_size.y; ++j) {
      for (auto k = 0; k < box_size.z; ++k) {
        auto x = make_float3(0.27f + initSpacing * j, 0.13f + initSpacing * i,
                             0.17f + initSpacing * k) +
                 particle_translation;
        pos.push_back(x);
      }
    }
  }
  auto granular_particles = std::make_shared<GranularParticles>(pos);

  std::vector<float3> upsampled_pos;
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-particle_radius,
                                                     particle_radius);

  for (const auto &particle : pos) {
    for (int n = 0; n < 15; ++n) {
      float3 offset =
          make_float3(distribution(generator), distribution(generator),
                      distribution(generator));
      upsampled_pos.push_back(particle + offset);
    }
  }

  auto upsampled_particles = std::make_shared<GranularParticles>(upsampled_pos);

  pos.clear();

  const auto compact_size = 2 * make_int3(ceil(space_size.x / cell_length),
                                          ceil(space_size.y / cell_length),
                                          ceil(space_size.z / cell_length));

  auto boundary_particles = std::make_shared<GranularParticles>(pos);
  bool is_move = false;

  if (scene == Scene::PILING) {

    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(i, 0, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, compact_size.y - 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }

    for (auto &p : pos) {
      p += boundary_translation;
    }

    boundary_particles = std::make_shared<GranularParticles>(pos);

  } else if (scene == Scene::BOX) {
    // front and back
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.y; ++j) {
        auto x = make_float3(i, j, 0) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, j, compact_size.z - 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(i, 0, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, compact_size.y - 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // left and right
    for (auto i = 0; i < compact_size.y - 2; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(0, i + 1, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(compact_size.x - 1, i + 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }

    for (auto &p : pos) {
      p += boundary_translation;
    }

    boundary_particles = std::make_shared<GranularParticles>(pos);
  } else if (scene == Scene::EXCAVATOR) {

    // add the box
    // front and back
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.y; ++j) {
        auto x = make_float3(i, j, 0) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, j, compact_size.z - 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(i, 0, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, compact_size.y - 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // left and right
    for (auto i = 0; i < compact_size.y - 2; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(0, i + 1, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(compact_size.x - 1, i + 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }

    // add the bucket
    //  Read additional boundary particles from file if specified
    int start_index = pos.size(); // Store the starting index

    int base_boundary_count = pos.size();
    std::vector<int> animated_markers(base_boundary_count, 0);

    try {
      if (std::filesystem::exists(obj_file)) {
        std::vector<float3> additional_particles =
            readBoundaryParticlesFromFile(obj_file,
                                          animation_state.start_index);

        // Store animation state
        animation_state.start_index = base_boundary_count;
        animation_state.num_particles = additional_particles.size();
        animation_state.is_animating = true;
        animation_state.animation_time = 0.0f;

        // Calculate rotation center
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        for (const auto &p : additional_particles) {
          centroid += p;
        }
        animation_state.rotation_center =
            centroid / (float)additional_particles.size();

        // Mark new particles as animated
        std::vector<int> new_animated(additional_particles.size(), 1);
        animated_markers.insert(animated_markers.end(), new_animated.begin(),
                                new_animated.end());

        for (auto &p : additional_particles) {
          p += make_float3(2.5f, 0.0f, 0.5f);
        }

        // Append new particles
        pos.insert(pos.end(), additional_particles.begin(),
                   additional_particles.end());
      }
    } catch (const std::exception &e) {
      std::cerr << "Error reading boundary particles: " << e.what()
                << std::endl;
    }

    boundary_particles = std::make_shared<GranularParticles>(pos);
    // Copy animation markers to device
    CUDA_CALL(cudaMemcpy(
        boundary_particles->get_is_animated_ptr(), animated_markers.data(),
        animated_markers.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Print some debug information
    std::cout << "Total boundary particles: " << pos.size() << std::endl;
    std::cout << "Animated particles: " << animation_state.num_particles
              << std::endl;
    std::cout << "Start index: " << animation_state.start_index << std::endl;

    is_move = true;

  } else if (scene == Scene::FUNNEL) {

    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.y; ++j) {
        auto x = make_float3(i, j, 0) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(i, j, compact_size.z - 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(i, 0, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        // x = make_float3(i, compact_size.y - 1, j + 1) /
        //     make_float3(compact_size - make_int3(1)) * space_size;
        // pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // left and right
    for (auto i = 0; i < compact_size.y - 2; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(0, i + 1, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        x = make_float3(compact_size.x - 1, i + 1, j + 1) /
            make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // for (auto &p : pos) {
    //   p += boundary_translation;
    // }

    int base_boundary_count = pos.size();
    std::vector<int> animated_markers(base_boundary_count, 0);
    try {
      if (std::filesystem::exists(obj_file)) {
        std::vector<float3> additional_particles =
            readBoundaryParticlesFromFile(obj_file,
                                          animation_state.start_index);

        animation_state.start_index = base_boundary_count;
        animation_state.num_particles = additional_particles.size();
        animation_state.is_animating = false;
        animation_state.animation_time = 0.0f;

        // Calculate rotation center
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        for (const auto &p : additional_particles) {
          centroid += p;
        }
        centroid = centroid / (float)additional_particles.size();

        // Apply 90-degree rotation around x-axis for each particle
        const float angle =
            90.0f * M_PI / 180.0f; // Convert 90 degrees to radians
        const float cos_theta = cosf(angle);
        const float sin_theta = sinf(angle);

        std::vector<int> new_animated(additional_particles.size(), 1);
        animated_markers.insert(animated_markers.end(), new_animated.begin(),
                                new_animated.end());

        for (auto &p : additional_particles) {
          // Translate to origin
          float3 centered = p - centroid;

          // Rotate around x-axis
          // For x-axis rotation:
          // y' = y*cos(θ) - z*sin(θ)
          // z' = y*sin(θ) + z*cos(θ)
          // x' = x
          float new_y = centered.y * cos_theta - centered.z * sin_theta;

          float new_z = centered.y * sin_theta + centered.z * cos_theta;

          // Store animation state

          // Translate back and apply any additional translation
          p = make_float3(centered.x, new_y, new_z) + centroid +
              make_float3(1.25f, 0.3f, 1.0f);
        }

        stored_additional_particles = additional_particles;
        // Append rotated particles
        pos.insert(pos.end(), additional_particles.begin(),
                   additional_particles.end());
      }
    } catch (const std::exception &e) {
      std::cerr << "Error reading boundary particles: " << e.what()
                << std::endl;
    }

    boundary_particles = std::make_shared<GranularParticles>(pos);

    CUDA_CALL(cudaMemcpy(
        boundary_particles->get_is_animated_ptr(), animated_markers.data(),
        animated_markers.size() * sizeof(int), cudaMemcpyHostToDevice));
  } else if (scene == Scene::CORNER) {

    // front and back
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.y; ++j) {
        auto x = make_float3(i, j, 0) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        // x = make_float3(i, j, compact_size.z - 1) /
        //     make_float3(compact_size - make_int3(1)) * space_size;
        // pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(i, 0, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        // x = make_float3(i, compact_size.y - 1, j + 1) /
        //     make_float3(compact_size - make_int3(1)) * space_size;
        // pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }
    // left and right
    for (auto i = 0; i < compact_size.y - 2; ++i) {
      for (auto j = 0; j < compact_size.z - 2; ++j) {
        auto x = make_float3(0, i + 1, j + 1) /
                 make_float3(compact_size - make_int3(1)) * space_size;
        pos.push_back(0.99f * x + 0.005f * space_size);
        // x = make_float3(compact_size.x - 1, i + 1, j + 1) /
        //     make_float3(compact_size - make_int3(1)) * space_size;
        // pos.push_back(0.99f * x + 0.005f * space_size);
      }
    }

    for (auto &p : pos) {
      p += boundary_translation;
    }

    boundary_particles = std::make_shared<GranularParticles>(pos);
  }

  p_system = std::make_shared<GranularSystem>(
      granular_particles, boundary_particles, upsampled_particles, space_size,
      cell_length, dt, G, cell_size, density, upsampled_particle_radius,
      is_move, is_adaptive);
}

__global__ void animateParticles(float3 *positions, int *is_animated,
                                 int num_particles, float3 translation,
                                 float rotation_angle, float3 rotation_center) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  // Only animate if this particle is marked for animation
  if (is_animated[idx] == 1) {
    float3 pos = positions[idx];

    // Apply translation
    pos += translation;

    // Apply rotation around z-axis
    if (rotation_angle != 0.0f) {
      // Translate to origin
      pos.x -= rotation_center.x;
      pos.y -= rotation_center.y;

      float angle_rad = rotation_angle * M_PI / 180.0f;
      float cos_theta = cosf(angle_rad);
      float sin_theta = sinf(angle_rad);

      float new_x = pos.x * cos_theta - pos.y * sin_theta;
      float new_y = pos.x * sin_theta + pos.y * cos_theta;

      pos.x = new_x + rotation_center.x;
      pos.y = new_y + rotation_center.y;
    }

    positions[idx] = pos;
  }
}

void updateAnimation(float dt) {
  if (!animation_state.is_animating)
    return;

  animation_state.animation_time += dt;

  float3 translation = make_float3(0.0f, 0.0f, 0.0f);
  float rotation = 0.0f;

  // First 5 seconds: translation
  if (animation_state.animation_time <= 2.5f &&
      !animation_state.translation_complete) {
    translation = animation_state.translation_direction *
                  animation_state.translation_speed * dt;
  } else if (!animation_state.translation_complete) {
    animation_state.translation_complete = true;
    animation_state.animation_time = 0.0f;

    // Recalculate rotation center when translation completes
    float3 *positions;
    int size = p_system->get_boundaries()->size();
    positions = new float3[size];
    CUDA_CALL(cudaMemcpy(positions, p_system->get_boundaries()->get_pos_ptr(),
                         size * sizeof(float3), cudaMemcpyDeviceToHost));

    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;
    for (int i = animation_state.start_index;
         i < animation_state.start_index + animation_state.num_particles; i++) {
      centroid += positions[i];
      count++;
    }
    if (count > 0) {
      animation_state.rotation_center = centroid / (float)count;
    }

    delete[] positions;
  }

  // After translation: rotation
  // Change from 90.0f to 45.0f for half the rotation
  if (animation_state.translation_complete &&
      animation_state.rotation_angle < 45.0f) { // Changed from 90.0f to 45.0f
    float rotation_delta = -animation_state.rotation_speed * dt;
    animation_state.rotation_angle += abs(rotation_delta);
    rotation = rotation_delta;

    if (animation_state.rotation_angle >=
        45.0f) { // Changed from 90.0f to 45.0f
      animation_state.is_animating = false;
    }
  }

  // Apply animation
  if (animation_state.is_animating) {
    const int block_size = 256;
    int num_blocks =
        (p_system->get_boundaries()->size() + block_size - 1) / block_size;

    animateParticles<<<num_blocks, block_size>>>(
        p_system->get_boundaries()->get_pos_ptr(),
        p_system->get_boundaries()->get_is_animated_ptr(),
        p_system->get_boundaries()->size(), translation, rotation,
        animation_state.rotation_center);

    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void resizeVBO(GLuint *vbo, size_t new_size) {
  // Unregister old VBO from CUDA
  CUDA_CALL(cudaGLUnregisterBufferObject(*vbo));

  // Delete old VBO
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glBufferData(GL_ARRAY_BUFFER, new_size * sizeof(float3), nullptr,
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Re-register with CUDA
  CUDA_CALL(cudaGLRegisterBufferObject(*vbo));
}

void createVBO(GLuint *vbo, const unsigned int size) {
  // Check if VBO already exists
  if (*vbo != 0) {
    // Unregister from CUDA if needed
    cudaError_t err = cudaGLUnregisterBufferObject(*vbo);
    if (err != cudaSuccess && err != cudaErrorInvalidResourceHandle) {
      std::cerr << "CUDA unregister error: " << cudaGetErrorString(err)
                << std::endl;
    }

    // Delete existing VBO
    glDeleteBuffers(1, vbo);
  }

  // create buffer object
  glGenBuffers(1, vbo);
  if (*vbo == 0) {
    throw std::runtime_error("Failed to create VBO");
  }

  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glBufferData(GL_ARRAY_BUFFER, size * sizeof(float3), nullptr,
               GL_DYNAMIC_DRAW);

  // Check for OpenGL errors
  GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    std::cerr << "OpenGL error in createVBO: " << gl_error << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    throw std::runtime_error("OpenGL error in createVBO");
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register buffer object with CUDA
  cudaError_t cuda_err = cudaGLRegisterBufferObject(*vbo);
  if (cuda_err != cudaSuccess) {
    std::cerr << "CUDA register error: " << cudaGetErrorString(cuda_err)
              << std::endl;
    throw std::runtime_error("Failed to register VBO with CUDA");
  }
}

void deleteVBO(GLuint *vbo) {
  if (*vbo == 0)
    return;

  // Unregister from CUDA
  cudaError_t err = cudaGLUnregisterBufferObject(*vbo);
  if (err != cudaSuccess && err != cudaErrorInvalidResourceHandle) {
    std::cerr << "CUDA unregister error: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Delete buffer
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glDeleteBuffers(1, vbo);

  *vbo = 0;
}

void onClose(void) {
  deleteVBO(&particlesVBO);
  deleteVBO(&particlesColorVBO);
  deleteVBO(&upsampledParticlesVBO);
  deleteVBO(&upsampledParticlesColorVBO);

  saveFrameTimes(p_system->get_frame_times());
  p_system = nullptr;
  CUDA_CALL(cudaDeviceReset());

  exit(0);
}

namespace particle_attributes {
enum {
  POSITION,
  COLOR,
  SIZE,
};
}

__global__ void markParticlesInSphere(float3 *positions, int num_particles,
                                      float3 sphere_center, float sphere_radius,
                                      int *picked_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  float3 pos = positions[idx];
  float3 diff = make_float3(pos.x - sphere_center.x, pos.y - sphere_center.y,
                            pos.z - sphere_center.z);
  float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

  // Print some particle positions for debugging
  if (idx < 5) { // Only print first 5 particles
    printf("Particle %d position: (%f, %f, %f)\n", idx, pos.x, pos.y, pos.z);
  }

  if (dist <= sphere_radius) {
    int picked_idx = atomicAdd(picked_indices, 1);
    if (picked_idx < num_particles) {
      picked_indices[picked_idx + 1] = idx;
      printf("Picked particle %d at position (%f, %f, %f), distance %f\n", idx,
             pos.x, pos.y, pos.z, dist);
    }
  }
}

__global__ void updatePickedParticlesPositions(float3 *positions,
                                               int *picked_indices,
                                               int num_picked, float3 delta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_picked)
    return;

  int particle_idx = picked_indices[idx];
  float3 old_pos = positions[particle_idx];
  float3 new_pos = old_pos + delta;

  // Add bounds checking (adjust these values based on your scene)
  new_pos.x = fmaxf(0.01f, fminf(new_pos.x, 0.99f));
  new_pos.y = fmaxf(0.01f, fminf(new_pos.y, 0.99f));
  new_pos.z = fmaxf(0.01f, fminf(new_pos.z, 0.99f));

  positions[particle_idx] = new_pos;

  // Debug print
  if (idx == 0) { // Only print for first particle to avoid spam
    printf("Particle %d moved from (%f, %f, %f) to (%f, %f, %f)\n",
           particle_idx, old_pos.x, old_pos.y, old_pos.z, new_pos.x, new_pos.y,
           new_pos.z);
  }
}

float3 screenToWorld(int x, int y) {
  GLint viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLfloat winX, winY, winZ;
  GLdouble posX, posY, posZ;

  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projection);

  // Convert screen coordinates to normalized device coordinates
  winX = (float)x;
  winY = (float)viewport[3] - (float)y;

  // Use a fixed depth for movement
  winZ = 0.95f; // Use a consistent depth for movement

  gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY,
               &posZ);

  // Scale to match particle space
  float3 world_pos =
      make_float3((float)posX * 0.1f, (float)posY * 0.1f, (float)posZ * 0.1f);

  return world_pos;
}

void mouseFunc(const int button, const int state, const int x, const int y) {
  if (GLUT_DOWN == state) {
    if (GLUT_LEFT_BUTTON == button) {
      mouse_left_down = true;
      mousePos[0] = x;
      mousePos[1] = y;
    } else if (GLUT_RIGHT_BUTTON == button) {
    }
  } else {
    mouse_left_down = false;
  }
  return;
}

void motionFunc(const int x, const int y) {
  int dx, dy;
  if (-1 == mousePos[0] && -1 == mousePos[1]) {
    mousePos[0] = x;
    mousePos[1] = y;
    dx = dy = 0;
  } else {
    dx = x - mousePos[0];
    dy = y - mousePos[1];
  }
  if (mouse_left_down) {
    rot[0] += (float(dy) * 180.0f) / 720.0f;
    rot[1] += (float(dx) * 180.0f) / 720.0f;
  }

  mousePos[0] = x;
  mousePos[1] = y;

  glutPostRedisplay();
  return;
}

extern "C" void
generate_dots(float3 *dot, float3 *color,
              const std::shared_ptr<GranularParticles> particles, int *surface,
              const float max_mass, const float min_mass, int show_surface);

void renderParticles(void) {
  size_t current_particle_count = p_system->size();

  // Check if particle count has changed
  if (current_particle_count != last_particle_count) {
    std::cout << "Particle count changed from " << last_particle_count << " to "
              << current_particle_count << std::endl;

    try {
      // Resize VBOs if needed
      resizeVBO(&particlesVBO, current_particle_count);
      resizeVBO(&particlesColorVBO, current_particle_count);
      last_particle_count = current_particle_count;
    } catch (const std::exception &e) {
      std::cerr << "Error resizing VBOs: " << e.what() << std::endl;
      return;
    }
  }

  // map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  float3 *cptr;

  cudaError_t err;

  err = cudaGLMapBufferObject((void **)&dptr, particlesVBO);
  if (err != cudaSuccess) {
    std::cerr << "Error mapping position VBO: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  err = cudaGLMapBufferObject((void **)&cptr, particlesColorVBO);
  if (err != cudaSuccess) {
    cudaGLUnmapBufferObject(particlesVBO); // Clean up first buffer
    std::cerr << "Error mapping color VBO: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  try {
    // calculate the dots' position and color
    generate_dots(dptr, cptr, p_system->get_particles(),
                  p_system->get_particles()->get_surface_ptr(),
                  p_system->get_max_mass(), p_system->get_min_mass(),
                  show_surface);
  } catch (const std::exception &e) {
    std::cerr << "Error in generate_dots: " << e.what() << std::endl;
  }

  // unmap buffer objects
  CUDA_CALL(cudaGLUnmapBufferObject(particlesVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(particlesColorVBO));

  glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, current_particle_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

void renderUpsampledParticles(void) {
  size_t current_upsampled_count = p_system->upsampled_size();

  // map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  float3 *cptr;

  CUDA_CALL(cudaGLMapBufferObject((void **)&dptr, upsampledParticlesVBO));
  CUDA_CALL(cudaGLMapBufferObject((void **)&cptr, upsampledParticlesColorVBO));

  try {
    // calculate the dots' position and color
    generate_dots(dptr, cptr, p_system->get_upsampled(),
                  p_system->get_upsampled()->get_surface_ptr(),
                  p_system->get_max_mass(), p_system->get_min_mass(),
                  show_surface);
  } catch (const std::exception &e) {
    std::cerr << "Error in generate_dots for upsampled particles: " << e.what()
              << std::endl;
  }

  // unmap buffer objects
  CUDA_CALL(cudaGLUnmapBufferObject(upsampledParticlesVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(upsampledParticlesColorVBO));

  glBindBuffer(GL_ARRAY_BUFFER, upsampledParticlesVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, upsampledParticlesColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, current_upsampled_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

void renderBoundaryCorners() {
  auto boundary_particles = p_system->get_boundaries();
  size_t boundary_count = boundary_particles->size();

  // Create VBOs for boundary corners if they don't exist
  static GLuint boundaryVBO = 0;
  static GLuint boundaryColorVBO = 0;

  if (boundaryVBO == 0) {
    createVBO(&boundaryVBO, sizeof(float3) * boundary_count);
    createVBO(&boundaryColorVBO, sizeof(float3) * boundary_count);
  }

  // Map OpenGL buffer objects for writing from CUDA
  float3 *dptr;
  float3 *cptr;

  CUDA_CALL(cudaGLMapBufferObject((void **)&dptr, boundaryVBO));
  CUDA_CALL(cudaGLMapBufferObject((void **)&cptr, boundaryColorVBO));

  // Copy positions from device to mapped buffer
  CUDA_CALL(cudaMemcpy(dptr, boundary_particles->get_pos_ptr(),
                       boundary_count * sizeof(float3),
                       cudaMemcpyDeviceToDevice));

  // Set all colors to red for corner particles
  std::vector<float3> colors(boundary_count, make_float3(1.0f, 0.0f, 0.0f));
  CUDA_CALL(cudaMemcpy(cptr, colors.data(), boundary_count * sizeof(float3),
                       cudaMemcpyHostToDevice));

  // Unmap buffer objects
  CUDA_CALL(cudaGLUnmapBufferObject(boundaryVBO));
  CUDA_CALL(cudaGLUnmapBufferObject(boundaryColorVBO));

  // Set up rendering state
  glPointSize(10.0f); // Larger points for corners

  // Enable vertex arrays
  glBindBuffer(GL_ARRAY_BUFFER, boundaryVBO);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, boundaryColorVBO);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  // Draw the boundary corners
  glDrawArrays(GL_POINTS, 0, boundary_count);

  // Cleanup state
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glPointSize(10.0f); // Reset point size

  // Delete VBOs in destructor or cleanup
  static bool first_run = true;
  if (first_run) {
    atexit([]() {
      if (boundaryVBO != 0)
        deleteVBO(&boundaryVBO);
      if (boundaryColorVBO != 0)
        deleteVBO(&boundaryColorVBO);
    });
    first_run = false;
  }
}

void one_step() {
  // saveUpsampledPositionsToVTK(p_system->get_upsampled(), frameId);

  // if (!stored_additional_particles.empty()) {
  //   std::filesystem::create_directory("additional_boundary_data");
  //   std::string boundary_filename =
  //       "additional_boundary_data/boundary_" + std::to_string(frameId) +
  //       ".vtk";
  //   saveAdditionalBoundaryParticlesToVTK(p_system->get_boundaries(),
  //                                        stored_additional_particles,
  //                                        boundary_filename);
  // }

  saveBoundaryParticlesToVTK(p_system->get_boundaries(),
                             p_system->get_upsampled(), frameId);

  ++frameId;
  p_system->step();
  // const auto milliseconds = p_system->step();
  // totalTime += milliseconds;
  // printf("Frame %d - %2.2f ms, avg time - %2.2f ms/frame (%3.2f FPS)\r",
  // 	frameId%10000, milliseconds, totalTime / float(frameId),
  // float(frameId)*1000.0f/totalTime);
}

void initGL(void) {
  // create VBOs
  createVBO(&particlesVBO, sizeof(float3) * p_system->size());
  createVBO(&particlesColorVBO, sizeof(float3) * p_system->size());
  // initiate shader program

  // New VBOs for upsampled particles
  createVBO(&upsampledParticlesVBO,
            sizeof(float3) * p_system->upsampled_size());
  createVBO(&upsampledParticlesColorVBO,
            sizeof(float3) * p_system->upsampled_size());

  m_particles_program = glCreateProgram();
  glBindAttribLocation(m_particles_program, particle_attributes::SIZE,
                       "pointSize");
  ShaderUtility::attachAndLinkProgram(
      m_particles_program,
      ShaderUtility::loadShaders("src/shaders/particles.vert",
                                 "src/shaders/particles.frag"));
  return;
}

static void displayFunc(void) {
  if (running) {
    updateAnimation(dt);
    one_step();
  }

  glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Text rendering code remains the same...

  // Save current matrices and attributes
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Disable depth testing and lighting for text
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);

  // Set text color (white for visibility)
  glColor3f(1.0f, 1.0f, 1.0f);

  // Position text
  glRasterPos2i(10, glutGet(GLUT_WINDOW_HEIGHT) - 50);

  // Create and render text
  std::string text = "Frame: " + std::to_string(frameId);
  text +=
      "  Particles: " + std::to_string(p_system->size()); // Add particle count
  for (const char &c : text) {
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, c);
  }

  // Restore matrices and attributes
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glPopAttrib();

  // Continue with regular 3D rendering
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Rest of your original display code...
  if (scene == Scene::PILING) {
    float3 translated_eye =
        make_float3(0, 0, 1.0f / zoom) + particle_translation;
    float3 translated_center = make_float3(0, 0, 0) + particle_translation;
    gluLookAt(translated_eye.x, translated_eye.y, translated_eye.z,
              translated_center.x, translated_center.y, translated_center.z, 0,
              1, 0);
  } else {
    gluLookAt(0, 0, 1.0f / zoom, 0, 0, 0, 0, 1, 0);
  }

  // Modify the camera setup to include the translation
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (scene == Scene::PILING) {
    // Apply the same translation as particles to the camera position
    float3 translated_eye =
        make_float3(0, 0, 1.0f / zoom) + particle_translation;
    float3 translated_center = make_float3(0, 0, 0) + particle_translation;
    gluLookAt(translated_eye.x, translated_eye.y,
              translated_eye.z, // Eye position
              translated_center.x, translated_center.y,
              translated_center.z, // Look at point
              0, 1, 0);            // Up vector
  } else {
    // Original camera setup for other scenes
    gluLookAt(0, 0, 1.0f / zoom, // Eye position
              0, 0, 0,           // Look at point
              0, 1, 0);          // Up vector
  }

  // Rest of the rendering code remains the same...
  glPushMatrix();
  glRotatef(rot[0], 1.0f, 0.0f, 0.0f);
  glRotatef(rot[1], 0.0f, 1.0f, 0.0f); // Draw a wire cube for reference
  glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  glLineWidth(2.0);
  glEnable(GL_MULTISAMPLE_ARB);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glutSolidCube(1.0);

  // ----------------------------------------------------------------
  // Render Particles
  // ----------------------------------------------------------------
  glUseProgram(m_particles_program);
  // Scale for point sprites

  float uniformVal = (m_window_h / 1920.0f) *
                     (1080.0f / tanf(m_fov * 0.5f * float(M_PI) / 180.0f));

  // float uniformVal = m_window_h / tanf(m_fov * 0.5f * float(M_PI) / 180.0f);
  glUniform1f(glGetUniformLocation(m_particles_program, "pointScale"),
              uniformVal);
  glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"),
              particle_radius);

  glEnable(GL_POINT_SPRITE_ARB);
  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);

  glPushMatrix();
  glTranslatef(-0.5f, -0.5f, -0.5f);
  renderParticles();

  // Render upsampled particles
  glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"),
              upsampled_particle_radius);
  renderUpsampledParticles();

  // renderBoundaryCorners();

  glPopMatrix();
  glPopMatrix();

  glutSwapBuffers();
  glutPostRedisplay();
}

void keyboardFunc(const unsigned char key, const int x, const int y) {
  float3 sphere_center;
  switch (key) {
  case '1':
    frameId = 0;
    totalTime = 0.0f;
    break;
  case ' ':
    running = !running;
    break;
  case ',':
    zoom *= 1.2f;
    break;
  case '.':
    zoom /= 1.2f;
    break;
  case 'q':
  case 'Q':
    onClose();
    break;
  case 'r':
  case 'R':
    rot[0] = rot[1] = 0;
    zoom = 0.3f;
    break;
  case 'N':
    void one_step();
    one_step();
    break;
  case 's':
  case 'S': {
    break;
  }
  case 'e':
  case 'E': {
    // Trigger explosion with force of 50.0f (adjust this value as needed)
    auto particles = p_system->get_particles_non_const();
    p_system->get_solver().trigger_explosion(particles, 50.0f);
    break;
  }
  default:
    break;
  }
}
void reshape(int width, int height) {
  if (height == 0)
    height = 1; // Prevent divide-by-zero errors

  // Update the viewport to match the window dimensions
  glViewport(0, 0, width, height);

  // Set up the projection matrix to preserve aspect ratio
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  float aspect = (float)width / (float)height; // Calculate aspect ratio
  gluPerspective(45.0, aspect, 0.1, 100.0);    // FOV, aspect ratio, near, far

  glMatrixMode(GL_MODELVIEW); // Return to modelview matrix
}

int main(int argc, char *argv[]) {
  try {
    SceneConfig config;
    try {
      config = loadSceneConfig("scenes/excavator.json");
    } catch (const std::exception &e) {
      std::cerr << "Error loading scene config: " << e.what() << std::endl;
      return 1;
    }

    space_size = config.space_size;
    dt = config.dt;
    G = config.G;
    sphSpacing = config.sphSpacing;
    initSpacing = config.initSpacing;
    smoothing_radius = config.smoothing_radius;
    cell_length = config.cell_length;
    cell_size = config.cell_size;
    density = config.density;
    box_size = config.box_size;
    boundary_translation = config.boundary_translation;
    particle_translation = config.particle_translation;
    scene = config.scene;
    obj_file = config.obj_file;
    is_adaptive = config.is_adaptive;
    save_file = config.save_file;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);

    glutInitWindowPosition(400, 0);
    // Get screen dimensions
    int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    // Calculate centered position
    int posX = (screenWidth - m_window_h) / 2;
    int posY = (screenHeight - m_window_h) / 2;

    // Create a window at the center of the screen
    glutInitWindowPosition(posX, posY);
    glutInitWindowSize(m_window_h, m_window_h);
    // glutInitWindowSize(m_window_h, m_window_h);
    glutCreateWindow("");

    glutFullScreen();
    glutDisplayFunc(&displayFunc);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(&keyboardFunc);
    glutMouseFunc(&mouseFunc);
    glutMotionFunc(&motionFunc);

    glewInit();
    ////////////////////
    init_granular_system();
    initGL();

    std::cout << "Instructions\n";
    std::cout << "Controls\n";
    std::cout << "Space - Start/Pause\n";
    std::cout << "Key N - One Step Forward\n";
    std::cout << "Key Q - Quit\n";
    std::cout << "Key R - Reset Viewpoint\n";
    std::cout << "Key , - Zoom In\n";
    std::cout << "Key . - Zoom Out\n";
    std::cout << "Mouse Drag - Change Viewpoint\n\n";
    ////////////////////
    glutMainLoop();
  } catch (...) {
    std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__
              << "\n";
  }
  return 0;
}

// thrust::fill(thrust::device, _buffer_num_surface_neighbors.addr(),
//              _buffer_num_surface_neighbors.addr() + num, 0);

// find_num_surface_neighbors<<<(num - 1) / block_size + 1, block_size>>>(
//     _buffer_num_surface_neighbors.addr(), particles->get_pos_ptr(),
//     particles->get_mass_ptr(), num, _cell_start_particle.addr(),
//     _cell_size, _cell_length, _density, _buffer_boundary.addr());
