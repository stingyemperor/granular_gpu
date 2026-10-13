#include "SceneConfig.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace {

const SceneType kAllScenes[] = {SceneType::Piling,    SceneType::Box,
                                SceneType::Excavator, SceneType::Funnel,
                                SceneType::Corner,    SceneType::BoxOpening};

SceneType parseSceneType(const json &j) {
  if (j.is_number_integer()) {
    const int id = j.get<int>();
    if (id < 0 || id > static_cast<int>(SceneType::BoxOpening)) {
      throw std::runtime_error("unknown scene id " + std::to_string(id));
    }
    return static_cast<SceneType>(id);
  }
  const auto name = j.get<std::string>();
  for (const auto type : kAllScenes) {
    if (name == sceneTypeName(type)) {
      return type;
    }
  }
  throw std::runtime_error("unknown scene \"" + name + "\"");
}

float3 readFloat3(const json &j, const char *key, float3 fallback) {
  if (!j.contains(key)) {
    return fallback;
  }
  const auto &v = j.at(key);
  return make_float3(v.value("x", fallback.x), v.value("y", fallback.y),
                     v.value("z", fallback.z));
}

int3 readInt3(const json &j, const char *key, int3 fallback) {
  if (!j.contains(key)) {
    return fallback;
  }
  const auto &v = j.at(key);
  return make_int3(v.value("x", fallback.x), v.value("y", fallback.y),
                   v.value("z", fallback.z));
}

} // namespace

const char *sceneTypeName(SceneType type) {
  switch (type) {
  case SceneType::Piling:
    return "piling";
  case SceneType::Box:
    return "box";
  case SceneType::Excavator:
    return "excavator";
  case SceneType::Funnel:
    return "funnel";
  case SceneType::Corner:
    return "corner";
  case SceneType::BoxOpening:
    return "box_opening";
  }
  return "unknown";
}

int3 SceneConfig::cellSize() const {
  return make_int3(ceil(space_size.x / cell_length),
                   ceil(space_size.y / cell_length),
                   ceil(space_size.z / cell_length));
}

std::string resolveProjectPath(const std::string &path) {
  const std::filesystem::path p(path);
  if (path.empty() || p.is_absolute()) {
    return path;
  }
  return (std::filesystem::path(GRANULAR_SOURCE_DIR) / p).string();
}

SceneConfig loadSceneConfig(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("unable to open scene file: " + path);
  }

  json j;
  try {
    file >> j;
  } catch (const json::exception &e) {
    throw std::runtime_error(path + ": " + e.what());
  }

  SceneConfig c;
  c.path = path;
  try {
    c.scene = parseSceneType(j.at("scene"));
    c.space_size = readFloat3(j, "space_size", c.space_size);
    c.dt = j.value("dt", c.dt);
    c.gravity = readFloat3(j, "gravity", c.gravity);
    c.density = j.value("density", c.density);
    c.cell_length = j.value("cell_length", c.cell_length);

    if (j.contains("particle_grid")) {
      c.particle_grid = readInt3(j, "particle_grid", c.particle_grid);
    } else if (j.contains("box_size")) {
      // Legacy key: box_size.x counted layers along y and box_size.y along x,
      // and fractional sizes were rounded up by the generating loop.
      const float3 legacy = readFloat3(j, "box_size", make_float3(10, 10, 10));
      c.particle_grid =
          make_int3(ceilf(legacy.y), ceilf(legacy.x), ceilf(legacy.z));
    }
    c.init_spacing = j.value("init_spacing", c.init_spacing);
    c.particle_origin = readFloat3(j, "particle_origin", c.particle_origin);
    c.particle_translation =
        readFloat3(j, "particle_translation", c.particle_translation);
    c.boundary_translation =
        readFloat3(j, "boundary_translation", c.boundary_translation);

    c.particle_radius = j.value("particle_radius", c.particle_radius);
    c.upsampled_radius = j.value("upsampled_radius", c.upsampled_radius);
    c.upsample_count = j.value("upsample_count", c.upsample_count);

    if (j.contains("adaptive")) {
      c.adaptive = j.at("adaptive").get<bool>();
    } else if (j.contains("is_adaptive")) {
      c.adaptive = j.at("is_adaptive").get<int>() != 0;
    }
    c.obj_file = resolveProjectPath(j.value("obj_file", std::string()));
    c.animate_obj = j.value("animate_obj", c.animate_obj);

    auto &out = c.output;
    if (j.contains("output")) {
      const auto &o = j.at("output");
      out.dir = o.value("dir", out.dir);
      out.stats_csv = o.value("stats_csv", out.stats_csv);
      out.frame_times = o.value("frame_times", out.frame_times);
      if (o.contains("export")) {
        const auto &e = o.at("export");
        auto &x = out.exports;
        x.upsampled = e.value("upsampled", x.upsampled);
        x.upsampled_centered =
            e.value("upsampled_centered", x.upsampled_centered);
        x.upsampled_csv = e.value("upsampled_csv", x.upsampled_csv);
        x.by_mass = e.value("by_mass", x.by_mass);
        x.animated_boundary = e.value("animated_boundary", x.animated_boundary);
        x.animated_boundary_centered =
            e.value("animated_boundary_centered", x.animated_boundary_centered);
        x.start_frame = e.value("start_frame", x.start_frame);
      }
    } else if (j.contains("save_file")) {
      out.dir = j.at("save_file").get<std::string>();
    }
    out.dir = resolveProjectPath(out.dir);
  } catch (const json::exception &e) {
    throw std::runtime_error(path + ": " + e.what());
  }

  if (c.cell_length <= 0.0f || c.dt <= 0.0f || c.upsample_count < 0) {
    throw std::runtime_error(path + ": cell_length and dt must be positive, "
                                    "upsample_count non-negative");
  }
  return c;
}
