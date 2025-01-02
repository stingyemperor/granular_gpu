#pragma once
#include "GranularParticles.hpp"
#include "Solver.hpp"
#include <memory>

class GranularSystem {
public:
  GranularSystem(std::shared_ptr<GranularParticles> &particles,
                 std::shared_ptr<GranularParticles> &boundary_particles,
                 float3 space_size, float cell_length, float dt, float3 g,
                 int3 cell_size, const int density);

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

  // TODO: add solver
  /**
   * @brief start index of the cell
   */
  DArray<int> _cell_start_particle;
  DArray<int> _cell_start_boundary;
  const float _cell_length;
  const float3 _space_size;
  const float _dt;
  const float3 _g;
  const int _density;
  const int3 _cell_size;
  /** \brief Device array to hold the cell index of each particle*/
  DArray<int> _buffer_int;
  DArray<int> _buffer_boundary;

  Solver _solver;

  void compute_boundary_mass();
  void neighbor_search(const std::shared_ptr<GranularParticles> &particles,
                       DArray<int> &cell_start);
  void
  set_surface_particles(const std::shared_ptr<GranularParticles> &particles,
                        DArray<int> &cell_start);
};
