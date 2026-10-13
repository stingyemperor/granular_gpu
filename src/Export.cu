#include "Export.hpp"
#include "helper_math.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool isValid(const float3 &p) {
  return !isnan(p.x) && !isnan(p.y) && !isnan(p.z);
}

template <typename T> std::vector<T> download(const T *device_ptr, size_t n) {
  std::vector<T> host(n);
  CUDA_CALL(cudaMemcpy(host.data(), device_ptr, n * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return host;
}

std::vector<float3> validPositions(const GranularParticles &particles,
                                   const char *what, int frame) {
  const auto all = download(particles.get_pos_ptr(), particles.size());
  std::vector<float3> valid;
  valid.reserve(all.size());
  for (const auto &p : all) {
    if (isValid(p)) {
      valid.push_back(p);
    }
  }
  if (valid.size() < all.size()) {
    std::cerr << "Warning: frame " << frame << " - skipped "
              << all.size() - valid.size() << " " << what
              << " particles with NaN positions" << std::endl;
  }
  return valid;
}

std::vector<float3> animatedBoundary(const GranularParticles &boundaries) {
  const auto pos = download(boundaries.get_pos_ptr(), boundaries.size());
  const auto flags =
      download(boundaries.get_is_animated_ptr(), boundaries.size());
  std::vector<float3> out;
  for (size_t i = 0; i < pos.size(); ++i) {
    if (flags[i] == 1) {
      out.push_back(pos[i]);
    }
  }
  return out;
}

// Shift that moves the centroid of `reference` to the origin and then lifts
// everything so the lowest reference point sits at y = 0.
float3 centeringOffset(const std::vector<float3> &reference) {
  if (reference.empty()) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  float3 c = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &p : reference) {
    c += p;
  }
  c = c / (float)reference.size();

  float min_y = 1000000.f;
  for (const auto &p : reference) {
    min_y = fminf(min_y, p.y - c.y);
  }
  const float y_offset = min_y < 0.0f ? -min_y : 0.0f;
  return make_float3(-c.x, -c.y + y_offset, -c.z);
}

std::string framePath(const SceneConfig &config, const std::string &mode,
                      int frame, const char *ext = ".vtk") {
  const fs::path dir = fs::path(config.output.dir) / mode;
  return (dir / (std::to_string(frame) + ext)).string();
}

std::ofstream openForWrite(const std::string &filename) {
  fs::create_directories(fs::path(filename).parent_path());
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
  }
  return out;
}

// Legacy VTK point cloud. y and z are swapped so "up" is +z in ParaView.
void writeVTKPoints(const std::string &filename, const char *title,
                    const std::vector<float3> &points,
                    float3 offset = make_float3(0.0f, 0.0f, 0.0f)) {
  if (points.empty()) {
    return;
  }
  std::ofstream out = openForWrite(filename);
  if (!out.is_open()) {
    return;
  }

  out << "# vtk DataFile Version 3.0\n"
      << title << "\nASCII\nDATASET UNSTRUCTURED_GRID\n";
  out << "POINTS " << points.size() << " float\n";
  for (const auto &point : points) {
    const float3 p = point + offset;
    out << p.x << " " << p.z << " " << p.y << "\n";
  }
  out << "CELLS " << points.size() << " " << points.size() * 2 << "\n";
  for (size_t i = 0; i < points.size(); i++) {
    out << "1 " << i << "\n";
  }
  out << "CELL_TYPES " << points.size() << "\n";
  for (size_t i = 0; i < points.size(); i++) {
    out << "1\n";
  }
}

void writeCSVPoints(const std::string &filename,
                    const std::vector<float3> &points) {
  std::ofstream out = openForWrite(filename);
  if (!out.is_open()) {
    return;
  }
  for (const auto &p : points) {
    out << p.x << "," << p.y << "," << p.z << "\n";
  }
}

// Coarse particles split into light (< 2), medium (2..3) and heavy (> 3).
void exportByMass(const SceneConfig &config, const GranularParticles &coarse,
                  int frame) {
  const auto pos = download(coarse.get_pos_ptr(), coarse.size());
  const auto mass = download(coarse.get_mass_ptr(), coarse.size());

  std::vector<float3> valid, light, medium, heavy;
  std::vector<float> valid_mass;
  for (size_t i = 0; i < pos.size(); ++i) {
    if (isValid(pos[i])) {
      valid.push_back(pos[i]);
      valid_mass.push_back(mass[i]);
    }
  }
  for (size_t i = 0; i < valid.size(); ++i) {
    if (valid_mass[i] < 2.0f) {
      light.push_back(valid[i]);
    } else if (valid_mass[i] <= 3.0f) {
      medium.push_back(valid[i]);
    } else {
      heavy.push_back(valid[i]);
    }
  }

  const float3 offset = centeringOffset(valid);
  writeVTKPoints(framePath(config, "mass_light", frame), "Particles by Mass",
                 light, offset);
  writeVTKPoints(framePath(config, "mass_medium", frame), "Particles by Mass",
                 medium, offset);
  writeVTKPoints(framePath(config, "mass_heavy", frame), "Particles by Mass",
                 heavy, offset);
}

} // namespace

void exportFrame(const SceneConfig &config, GranularSystem &system, int frame) {
  const auto &x = config.output.exports;
  if (!x.any() || frame < x.start_frame) {
    return;
  }

  const auto &upsampled = *system.get_upsampled();
  const auto &boundaries = *system.get_boundaries();
  const std::string suffix = config.adaptive ? "_adaptive" : "_normal";

  std::vector<float3> fine;
  if (x.upsampled || x.upsampled_centered || x.animated_boundary_centered) {
    fine = validPositions(upsampled, "upsampled", frame);
  }

  if (x.upsampled) {
    writeVTKPoints(framePath(config, "upsampled" + suffix, frame),
                   "Upsampled Granular Particles", fine);
  }
  if (x.upsampled_centered) {
    writeVTKPoints(framePath(config, "upsampled_centered" + suffix, frame),
                   "Upsampled Granular Particles", fine, centeringOffset(fine));
  }
  if (x.upsampled_csv) {
    writeCSVPoints(framePath(config, "upsampled_csv" + suffix, frame, ".csv"),
                   download(upsampled.get_pos_ptr(), upsampled.size()));
  }
  if (x.by_mass) {
    exportByMass(config, *system.get_particles(), frame);
  }
  if (x.animated_boundary || x.animated_boundary_centered) {
    const auto obj = animatedBoundary(boundaries);
    if (x.animated_boundary) {
      writeVTKPoints(framePath(config, "obj_boundary", frame),
                     "Animated Boundary Particles", obj);
    }
    if (x.animated_boundary_centered) {
      writeVTKPoints(framePath(config, "obj_boundary_centered", frame),
                     "Animated Boundary Particles", obj, centeringOffset(fine));
    }
  }
}

void saveRunStats(const SceneConfig &config, GranularSystem &system) {
  const auto &out = config.output;
  if (!out.stats_csv && !out.frame_times) {
    return;
  }
  fs::create_directories(out.dir);
  if (out.stats_csv) {
    system.get_solver().save_particle_stats(
        (fs::path(out.dir) / "stats.csv").string());
  }
  if (out.frame_times) {
    const auto path = fs::path(out.dir) / "frame_times.txt";
    std::ofstream file(path);
    for (float t : system.get_frame_times()) {
      file << t << "\n";
    }
  }
}
