#pragma once
#include "GranularSystem.hpp"
#include "SceneConfig.hpp"

// Writes the per-frame files enabled in config.output.exports.
void exportFrame(const SceneConfig &config, GranularSystem &system, int frame);

// Writes stats.csv and frame_times.txt to config.output.dir (if enabled).
void saveRunStats(const SceneConfig &config, GranularSystem &system);
