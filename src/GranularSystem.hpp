#pragma once
#include "GranularParticles.hpp"
#include "Solver.hpp"
#include <memory>

class GranularSystem {
public:
  GranularSystem(std::shared_ptr<GranularParticles> &particles,
                 std::shared_ptr<GranularParticles> &boundary_particles,
                 std::shared_ptr<GranularParticles> &upsampled_particles,
                 float3 space_size, float cell_length, float dt, float3 g,
                 int3 cell_size, const float density,
                 const float upsampled_radius, const bool is_move_boundary);

  GranularSystem(const GranularSystem &) = delete;
  GranularSystem &operator=(const GranularSystem &) = delete;

  float step();

  int size() const { return (*_particles).size(); }
  int upsampled_size() const { return (*_upsampled).size(); }

  int boundary_size() const { return (*_boundaries).size(); }
  int total_size() const {
    return (*_particles).size() + (*_boundaries).size() + (*_upsampled).size();
  }

  auto get_particles() const {
    return static_cast<const std::shared_ptr<GranularParticles>>(_particles);
  }

  auto get_particles_non_const() const {
    return static_cast<std::shared_ptr<GranularParticles>>(_particles);
  }

  auto get_boundaries() const {
    return static_cast<const std::shared_ptr<GranularParticles>>(_boundaries);
  }
  auto get_upsampled() const {
    return static_cast<const std::shared_ptr<GranularParticles>>(_upsampled);
  }

  float const get_max_mass() { return _max_mass; }
  float const get_min_mass() { return _min_mass; }
  float const get_upsampled_radius() { return _upsampled_radius; }

  const std::vector<float> &get_frame_times() const { return frame_times; }

  ~GranularSystem() {

    // Force synchronization
    cudaDeviceSynchronize();
  }

  Solver &get_solver() { return _solver; }

private:
  std::shared_ptr<GranularParticles> _particles;
  const std::shared_ptr<GranularParticles> _boundaries;
  std::shared_ptr<GranularParticles> _upsampled;

  /**
   * @brief start index of the cell
   */
  DArray<int> _cell_start_particle;
  DArray<int> _cell_start_boundary;
  DArray<int> _cell_start_upsampled;
  const int _upsampled_dim;
  const float _cell_length;
  const float3 _space_size;
  const float _dt;
  const float3 _g;
  const float _density;
  const int _max_mass;
  const int _min_mass;
  const int3 _cell_size;
  const float _upsampled_radius;
  /** \brief Device array to hold the cell index of each particle*/
  DArray<int> _buffer_int;
  DArray<int> _buffer_boundary;
  DArray<float3> _buffer_cover_vector;
  DArray<int> _buffer_num_surface_neighbors;
  std::vector<float> frame_times; // Store frame times
  const bool _is_move_boundary;

  Solver _solver;

  void compute_boundary_mass();
  void
  neighbor_search_granular(const std::shared_ptr<GranularParticles> &particles,
                           DArray<int> &cell_start);
  void
  neighbor_search_boundary(const std::shared_ptr<GranularParticles> &particles,
                           DArray<int> &cell_start);
  void
  neighbor_search_upsampled(const std::shared_ptr<GranularParticles> &particles,
                            DArray<int> &cell_start);

  void
  set_surface_particles(const std::shared_ptr<GranularParticles> &particles,
                        DArray<int> &cell_start);
};
