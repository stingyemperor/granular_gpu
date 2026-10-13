#pragma once
#include <cuda_runtime.h>
#include <string>

enum class SceneType {
  Piling = 0,
  Box = 1,
  Excavator = 2,
  Funnel = 3,
  Corner = 4,
  BoxOpening = 5,
};

const char *sceneTypeName(SceneType type);

// Which per-frame files to write. Each mode writes <output.dir>/<mode>/<frame>.
struct ExportOptions {
  bool upsampled = false;          // upsampled particles, raw coordinates
  bool upsampled_centered = false; // upsampled particles, recentered on floor
  bool upsampled_csv = false;      // upsampled particles as CSV
  bool by_mass = false;            // coarse particles split into mass classes
  bool animated_boundary = false;  // obj boundary particles, raw coordinates
  bool animated_boundary_centered = false; // ... recentered like upsampled
  int start_frame = 0;

  bool any() const {
    return upsampled || upsampled_centered || upsampled_csv || by_mass ||
           animated_boundary || animated_boundary_centered;
  }
};

struct OutputOptions {
  std::string dir = "output";
  bool stats_csv = true;   // merge/split counts per adaptive step
  bool frame_times = true; // simulation time per frame (ms)
  ExportOptions exports;
};

struct SceneConfig {
  std::string path; // file this config was loaded from

  SceneType scene = SceneType::Box;
  float3 space_size = make_float3(1.8f, 1.8f, 1.8f);
  float dt = 0.002f;
  float3 gravity = make_float3(0.0f, -9.8f, 0.0f);
  float density = 238732.4146f;
  float cell_length = 0.0404f;

  // Initial block of granular particles: particle_grid.{x,y,z} particles
  // along each axis, init_spacing apart, starting at particle_origin.
  int3 particle_grid = make_int3(10, 10, 10);
  float init_spacing = 0.03f;
  float3 particle_origin = make_float3(0.27f, 0.13f, 0.17f);
  float3 particle_translation = make_float3(0.0f, 0.0f, 0.0f);
  float3 boundary_translation = make_float3(0.0f, 0.0f, 0.0f);

  float particle_radius = 0.01f;   // coarse particles (also upsample jitter)
  float upsampled_radius = 0.005f; // fine particles used for rendering
  int upsample_count = 25;         // fine particles per coarse particle

  bool adaptive = true;     // adaptive merging/splitting of coarse particles
  std::string obj_file;     // extra boundary particles (excavator, funnel)
  bool animate_obj = false; // move the obj boundary (excavator)

  OutputOptions output;

  int3 cellSize() const;
};

// Loads a scene file. Missing optional keys keep the defaults above.
SceneConfig loadSceneConfig(const std::string &path);

// Relative paths inside scene files (obj_file, output.dir) are resolved
// against the project root so the program can be run from any directory.
std::string resolveProjectPath(const std::string &path);
