#include "CUDAFunctions.cuh"
#include "System.hpp"
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector_functions.h>

System::System(std::shared_ptr<ParticleData> granular_particles,
               std::shared_ptr<ParticleData> boundary_particles,
               const float3 space_size, const float cell_length, const float dt,
               const float3 g, const int3 cell_size, const int density)
    : _granular_particles(granular_particles),
      _boundary_particles(boundary_particles), _space_size(space_size),
      _cell_length(cell_length), _dt(dt), _g(g), _cell_size(cell_size),
      _density(density), _max_mass(3), _min_mass(1) {

  _cell_start_granular.reserve(cell_size.x * cell_size.y * cell_size.z + 1);
  _cell_start_boundary.reserve(cell_size.x * cell_size.y * cell_size.z + 1);
  _cell_index_t.reserve(
      std::max(total_size(), cell_size.x * cell_size.y * cell_size.z + 1));

  thrust::fill(
      thrust::device, _granular_particles->get_mass().begin(),
      _granular_particles->get_mass().begin() + _granular_particles->size(), 1);
  step();

  _surface.reserve(_granular_particles->size());
}

// System::~System() {
//   // Clear device vectors explicitly
//   _surface.clear();
//   _cell_start_granular.clear();
//   _cell_start_boundary.clear();
//   _boundary_t.clear();
//   _cell_index_t.clear();
//
//   // Optionally shrink capacity to free memory
//   // _surface.shrink_to_fit();
//   // _cell_start_granular.shrink_to_fit();
//   // _cell_start_boundary.shrink_to_fit();
//   // _boundary_t.shrink_to_fit();
//   // _cell_index_t.shrink_to_fit();
//
//   // No need to manually delete shared_ptr resources;
//   // They are automatically managed by std::shared_ptr.
// }

void System::step() {};
