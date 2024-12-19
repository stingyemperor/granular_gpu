#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "CUDAFunctions.cuh"
#include "DArray.hpp"
#include "Particles.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"

GranularSystem::GranularSystem(
    std::shared_ptr<GranularParticles> &granular_particles,
    std::shared_ptr<GranularParticles> &boundary_particles,
    // TODO add solver
    const float3 space_size,
    const float dt,
    const float sphCellLength,
    int3 cell_size)
    : _particles(std::move(granular_particles)),
      _boundaries(std::move(boundary_particles)),
      _space_size(space_size),
      _dt(dt),
      cell_start_fluid(cell_size.x * cell_size.y * cell_size.z + 1),
      cell_start_boundary(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_size(cell_size),
      buffer_int(max(total_size(), cell_size.x * cell_size.y * cell_size.z + 1))
{
    neighbor_search(_boundaries, cell_start_boundary);
    compute_boundary_mass();

    thrust::fill(thrust::device, _particles->get_mass_ptr(), _particles->get_mass_ptr() + _particles->size(), 1);
    neighbor_search(_particles, cell_start_fluid);

    step();
}
