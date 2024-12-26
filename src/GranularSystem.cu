#include "CUDAFunctions.cuh"
#include "DArray.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

GranularSystem::GranularSystem(
    std::shared_ptr<GranularParticles> &granular_particles,
    std::shared_ptr<GranularParticles> &boundary_particles,
    const float3 space_size, const float cell_length, const float dt,
    int3 cell_size)
    : _particles(std::move(granular_particles)),
      _boundaries(std::move(boundary_particles)), _solver(_particles),
      _space_size(space_size), _dt(dt), _cell_length(cell_length),
      _cell_start_particle(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_start_boundary(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_size(cell_size),
      _buffer_int(
          std::max(total_size(), cell_size.x * cell_size.y * cell_size.z + 1)) {
  neighbor_search(_boundaries, _cell_start_boundary);
  compute_boundary_mass();

  // Set the mass of all the particles to 1
  thrust::fill(thrust::device, _particles->get_mass_ptr(),
               _particles->get_mass_ptr() + _particles->size(), 1);
  neighbor_search(_particles, _cell_start_particle);

  step();
}

void GranularSystem::neighbor_search(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cellStart) {
  int num = particles->size();

  // map the particles to their cell idx
  mapParticles2Cells_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      particles->get_particle_2_cell(), particles->get_pos_ptr(), _cell_length,
      _cell_size, num);
  // copy the cell indexes to _buffer_int
  CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                       sizeof(int) * num, cudaMemcpyDeviceToDevice));
  // sort the position with the cell indexes
  thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                      _buffer_int.addr() + num, particles->get_pos_ptr());
  // copy the new sorted indexes back to _buffer_int
  CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                       sizeof(int) * num, cudaMemcpyDeviceToDevice));
  // sort velocity based on the keys
  thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                      _buffer_int.addr() + num, particles->get_vel_ptr());

  // fill cell_start with zeroes
  thrust::fill(
      thrust::device, cellStart.addr(),
      cellStart.addr() + _cell_size.x * _cell_size.y * _cell_size.z + 1, 0);

  // add number of particles per cell index to cell_start
  countingInCell_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      cellStart.addr(), particles->get_particle_2_cell(), num);
  // calculate the prefix sum of cell_start to help with neighbor search
  thrust::exclusive_scan(thrust::device, cellStart.addr(),
                         cellStart.addr() +
                             _cell_size.x * _cell_size.y * _cell_size.z + 1,
                         cellStart.addr());
  return;
}

float GranularSystem::step() {
  // cudaEvent_t start, stop;
  // CUDA_CALL(cudaEventCreate(&start));
  // CUDA_CALL(cudaEventCreate(&stop));
  // CUDA_CALL(cudaEventRecord(start, 0));

  neighbor_search(_particles, _cell_start_particle);
  // try {
  // 	_solver->step(_fluids, _boundaries, cellStartFluid, cellStartBoundary,
  // 		_spaceSize, _cellSize, _sphCellLength, _sphSmoothingRadius,
  // 		_dt, _sphRho0, _sphRhoBoundary, _sphStiff, _sphVisc, _sphG,
  // 		_sphSurfaceTensionIntensity, _sphAirPressure);
  // 	cudaDeviceSynchronize(); CHECK_KERNEL();
  // }
  // catch (const char* s) {
  // 	std::cout << s << "\n";
  // }
  // catch (...) {
  // 	std::cout << "Unknown Exception at "<<__FILE__<<": line "<<__LINE__ <<
  // "\n";
  // }

  // float milliseconds;
  // CUDA_CALL(cudaEventRecord(stop, 0));
  // CUDA_CALL(cudaEventSynchronize(stop));
  // CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  // CUDA_CALL(cudaEventDestroy(start));
  // CUDA_CALL(cudaEventDestroy(stop));
  // return milliseconds;
  return 1;
}

void GranularSystem::compute_boundary_mass() {
  // computeBoundaryMass_CUDA<<<(_boundaries->size() - 1) / block_size + 1,
  //                            block_size>>>(
  //     _boundaries->getMassPtr(), _boundaries->getPosPtr(),
  //     _boundaries->size(), cellStartBoundary.addr(), _cellSize,
  //     _sphCellLength, _sphRhoBoundary, _sphSmoothingRadius);
}
