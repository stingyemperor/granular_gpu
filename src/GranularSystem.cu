#include "CUDAFunctions.cuh"
#include "DArray.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector_functions.h>

GranularSystem::GranularSystem(
    std::shared_ptr<GranularParticles> &granular_particles,
    std::shared_ptr<GranularParticles> &boundary_particles,
    const float3 space_size, const float cell_length, const float dt,
    const float3 g, int3 cell_size, const float density)
    : _particles(std::move(granular_particles)),
      _boundaries(std::move(boundary_particles)), _solver(_particles),
      _space_size(space_size), _dt(dt), _g(g), _cell_length(cell_length),
      _cell_start_particle(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_start_boundary(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_size(cell_size),
      _buffer_int(
          std::max(total_size(), cell_size.x * cell_size.y * cell_size.z + 1)),
      _density(density), _max_mass(4), _min_mass(1),
      _buffer_boundary(_particles->size()) {
  // initalize the boundary_particles
  neighbor_search(_boundaries, _cell_start_boundary);
  // Set the mass of all the particles to 1
  thrust::fill(thrust::device, _particles->get_mass_ptr(),
               _particles->get_mass_ptr() + _particles->size(), 1);
  neighbor_search(_particles, _cell_start_particle);

  step();
}

void GranularSystem::neighbor_search(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {

  int num = particles->size();
  std::cout << "Starting neighbor search for " << num << " particles"
            << std::endl;

  // Debug check array sizes
  std::cout << "Buffer int size: " << _buffer_int.length() << std::endl;
  std::cout << "Cell start size: " << cell_start.length() << std::endl;

  // Verify memory state
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "GPU Memory - Free: " << free_mem / 1024 / 1024
  //           << "MB, Total: " << total_mem / 1024 / 1024 << "MB" << std::endl;

  // Debug sync point before kernel launch
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Pre-kernel sync error: " << cudaGetErrorString(err)
              << std::endl;
    throw std::runtime_error("CUDA sync error before mapParticles2Cells");
  }

  // map the particles to their cell idx
  mapParticles2Cells_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      particles->get_particle_2_cell(), particles->get_pos_ptr(), _cell_length,
      _cell_size, num);

  // Check for kernel errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(err)
              << std::endl;
    throw std::runtime_error("CUDA kernel error in mapParticles2Cells");
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel sync error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA sync error after mapParticles2Cells");
  }

  // copy the cell indexes to _buffer_int with error checking
  try {
    CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                         sizeof(int) * num, cudaMemcpyDeviceToDevice));
  } catch (const std::exception &e) {
    std::cerr << "Error copying to buffer_int: " << e.what() << std::endl;
    throw;
  }

  // sort the position with the cell indexes
  try {
    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num, particles->get_pos_ptr());
  } catch (const std::exception &e) {
    std::cerr << "Error in sort_by_key: " << e.what() << std::endl;
    throw;
  }

  // copy the new sorted indexes back to _buffer_int
  try {
    CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                         sizeof(int) * num, cudaMemcpyDeviceToDevice));
  } catch (const std::exception &e) {
    std::cerr << "Error copying back to buffer_int: " << e.what() << std::endl;
    throw;
  }

  // sort velocity based on the keys
  try {
    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num, particles->get_vel_ptr());
  } catch (const std::exception &e) {
    std::cerr << "Error in velocity sort_by_key: " << e.what() << std::endl;
    throw;
  }

  // sort mass based on the keys
  try {
    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num, particles->get_mass_ptr());
  } catch (const std::exception &e) {
    std::cerr << "Error in mass sort_by_key: " << e.what() << std::endl;
    throw;
  }

  // sort solver buffers based on the keys
  try {

    _solver.resize(particles->size());

    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num,
                        _solver.get_buffer_merge_ptr());

    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num,
                        _solver.get_buffer_merge_count_ptr());

    thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                        _buffer_int.addr() + num,
                        _solver.get_buffer_remove_ptr());
  } catch (const std::exception &e) {
    std::cerr << "Error in mass sort_by_key: " << e.what() << std::endl;
    throw;
  }

  // fill cell_start with zeroes
  try {
    thrust::fill(
        thrust::device, cell_start.addr(),
        cell_start.addr() + _cell_size.x * _cell_size.y * _cell_size.z + 1, 0);
  } catch (const std::exception &e) {
    std::cerr << "Error filling cell_start: " << e.what() << std::endl;
    throw;
  }

  // add number of particles per cell index to cell_start
  countingInCell_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      cell_start.addr(), particles->get_particle_2_cell(), num);

  // Check for kernel errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "countingInCell kernel error: " << cudaGetErrorString(err)
              << std::endl;
    throw std::runtime_error("CUDA kernel error in countingInCell");
  }

  // calculate the prefix sum of cell_start
  try {
    thrust::exclusive_scan(thrust::device, cell_start.addr(),
                           cell_start.addr() +
                               _cell_size.x * _cell_size.y * _cell_size.z + 1,
                           cell_start.addr());
  } catch (const std::exception &e) {
    std::cerr << "Error in exclusive_scan: " << e.what() << std::endl;
    throw;
  }

  // Final sync point
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Final sync error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA sync error at end of neighbor_search");
  }
}

float GranularSystem::step() {
  // cudaEvent_t start, stop;
  // CUDA_CALL(cudaEventCreate(&start));
  // CUDA_CALL(cudaEventCreate(&stop));
  // CUDA_CALL(cudaEventRecord(start, 0));

  neighbor_search(_particles, _cell_start_particle);
  try {
    _solver.step(_particles, _boundaries, _cell_start_particle,
                 _cell_start_boundary, _space_size, _cell_size, _cell_length,
                 _dt, _g, _density);
    // cudaDeviceSynchronize();
    CHECK_KERNEL();
    set_surface_particles(_particles, _cell_start_particle);

    cudaDeviceSynchronize();

    _solver.adaptive_sampling(_particles, _cell_start_particle, _max_mass,
                              _cell_size, _space_size, _cell_length, _density);

  } catch (const char *s) {
    std::cout << s << "\n";
  } catch (...) {
    std::cout << "Unknown Exception at " << __FILE__ << ": line" << __LINE__
              << "\n";
  }

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
  //     _boundaries->size(), cell_startBoundary.addr(), _cell_size,
  //     _sphCellLength, _sphRhoBoundary, _sphSmoothingRadius);
}

// __device__ void boundary_kernel(float *sum_kernel, const int i,
//                                 const int cell_id, float3 *pos, int
//                                 *cell_start, const int3 cell_size, const
//                                 float density) {
//   if (cell_id == (cell_size.x * cell_size.y * cell_size.z))
//     return;
//   auto j = cell_start[cell_id];
//   const auto end = cell_start[cell_id + 1];
//   while (j < end) {
//     *sum_kernel += cubic_spline_kernel(length(pos[i] - pos[j]), radius);
//     ++j;
//   }
//   return;
// }
//
// __global__ void computeBoundaryMass_CUDA(float *mass, float3 *pos,
//                                          const int num, int *cell_start,
//                                          const int3 cell_size,
//                                          const float cell_length,
//                                          const float rhoB, const float
//                                          radius) {
//   const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//   if (i >= num)
//     return;
//   const auto cell_pos = make_int3(pos[i] / cell_length);
// #pragma unroll
//   for (auto m = 0; m < 27; ++m) {
//     const auto cellID = particlePos2cellIdx(
//         cell_pos + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
//         cell_size);
//     boundary_kernel(&mass[i], i, cellID, pos, cell_start, cell_size, radius);
//   }
//   mass[i] = rhoB / fmaxf(EPSILON, mass[i]);
//   return;
// }

__global__ void find_surface(int *buffer_boundary, float3 *pos_granular,
                             const int num, int *cell_start_granular,
                             const int3 cell_size, const float cell_length) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (i >= num)
    return;
  __syncthreads();

  float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
  unsigned int n_neighbors = 0;

  for (auto m = 0; m < 27; __syncthreads(), ++m) {
    const auto cellID = particlePos2cellIdx(
        make_int3(pos_granular[i] / cell_length) +
            make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
        cell_size);
    if (cellID == (cell_size.x * cell_size.y * cell_size.z))
      continue;
    // calculate centroid of neighbors
    int j = cell_start_granular[cellID];
    while (j < cell_start_granular[cellID + 1]) {
      if (i == j) {
        j++;
        continue;
      }
      centroid += pos_granular[j];
      n_neighbors++;
      j++;
    }
  }

  centroid /= (float)n_neighbors;

  float dis =
      norm3df(pos_granular[i].x - centroid.x, pos_granular[i].y - centroid.y,
              pos_granular[i].z - centroid.z);

  if (dis > 0.01 || n_neighbors < 5) {
    buffer_boundary[i] = 1;
  } else {
    buffer_boundary[i] = 0;
  }
}

void GranularSystem::set_surface_particles(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {
  const int num = particles->size();
  find_surface<<<(num - 1) / block_size + 1, block_size>>>(
      _buffer_boundary.addr(), particles->get_pos_ptr(), num,
      _cell_start_particle.addr(), _cell_size, _cell_length);

  CUDA_CALL(cudaMemcpy(particles->get_surface_ptr(), _buffer_boundary.addr(),
                       sizeof(int) * particles->size(),
                       cudaMemcpyDeviceToDevice));
}
