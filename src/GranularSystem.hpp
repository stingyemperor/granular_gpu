#pragma once
#include "GranularParticles.hpp"

class GranularSystem {
public:
  GranularSystem(std::shared_ptr<GranularParticles> &particles,
                 std::shared_ptr<GranularParticles> &boundary_particles,
                 // TODO add solver
                 float3 spaceSize, float cell_length, float dt, int3 cell_size);

  GranularSystem(const GranularSystem &) = delete;
  GranularSystem &operator=(const GranularSystem &) = delete;

  float step();

  int size() const { return (*_particles).size(); }

  int boundary_size() const { return (*_boundaries).size(); }
  int total_size() const {
    return (*_particles).size() + (*_boundaries).size();
  }

  auto get_particles() const {
    return static_cast<const std::shared_ptr<GranularParticles>>(_particles);
  }
  auto get_boundaries() const {
    return static_cast<const std::shared_ptr<GranularParticles>>(_boundaries);
  }
  ~GranularSystem() noexcept {}

private:
  std::shared_ptr<GranularParticles> _particles;
  const std::shared_ptr<GranularParticles> _boundaries;

  // TODO add solver
  DArray<int> _cell_start_fluid;
  DArray<int> _cell_start_boundary;
  const float _cell_length;
  const float3 _space_size;
  const float _dt;
  const int3 _cell_size;
  DArray<int> _buffer_int;

  void compute_boundary_mass();
  void neighbor_search(const std::shared_ptr<GranularParticles> &particles,
                       DArray<int> &cell_start);
};
