#include "CUDAFunctions.cuh"
#include "DArray.hpp"
#include "GranularParticles.hpp"
#include "GranularSystem.hpp"
#include "helper_math.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector_functions.h>
#include <vector_types.h>

#define SMOOTHING_LENGTH 0.04f
#define PI_3 1.047197551f
#define PI_2 1.570796327f
#define PI_2_2 0.7853981634
#define PI_3_2 0.5235987756
#define PI_3_4 0.2617993878
#define NEIGHBOR_THRESHOLD 13
#define MAX_NEIGHBORS 100
#define FLT_MAX 100000.0f

__device__ __constant__ double PI_d = 3.14159265358979323846;

GranularSystem::GranularSystem(
    std::shared_ptr<GranularParticles> &granular_particles,
    std::shared_ptr<GranularParticles> &boundary_particles,
    std::shared_ptr<GranularParticles> &upsampled_particles,
    const float3 space_size, const float cell_length, const float dt,
    const float3 g, int3 cell_size, const float density,
    const float upsampled_radius, const bool is_move_boundary,
    const int is_adaptive)
    : _particles(std::move(granular_particles)),
      _boundaries(std::move(boundary_particles)),
      _upsampled(std::move(upsampled_particles)), _solver(_particles),
      _upsampled_dim(15), _space_size(space_size), _dt(dt), _g(g),
      _cell_length(cell_length),
      _cell_start_particle(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_start_boundary(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_start_upsampled(cell_size.x * cell_size.y * cell_size.z + 1),
      _cell_size(cell_size),
      _buffer_int(
          std::max(total_size(), cell_size.x * cell_size.y * cell_size.z + 1)),
      _density(density), _max_mass(6), _min_mass(1),
      _upsampled_radius(upsampled_radius), _buffer_boundary(_particles->size()),
      _buffer_cover_vector(_particles->size()),
      _buffer_num_surface_neighbors(_particles->size()),
      _is_move_boundary(is_move_boundary), _is_adaptive(is_adaptive) {
  // initalize the boundary_particles
  neighbor_search_boundary(_boundaries, _cell_start_boundary);
  // Set the mass of all the particles to 1
  thrust::fill(thrust::device, _particles->get_mass_ptr(),
               _particles->get_mass_ptr() + _particles->size(), 1);

  neighbor_search_granular(_particles, _cell_start_particle);
  neighbor_search_upsampled(_upsampled, _cell_start_upsampled);
  step();
}

void GranularSystem::neighbor_search_granular(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {
  int num = particles->size();

  // Create and initialize indices array
  DArray<int> particle_indices(num);
  thrust::sequence(thrust::device, particle_indices.addr(),
                   particle_indices.addr() + num);

  // Map particles to cells
  mapParticles2Cells_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      particles->get_particle_2_cell(), particles->get_pos_ptr(), _cell_length,
      _cell_size, num);

  // Copy cell indices for sorting
  CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                       sizeof(int) * num, cudaMemcpyDeviceToDevice));

  // Sort positions and indices by cell
  thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                      _buffer_int.addr() + num,
                      thrust::make_zip_iterator(thrust::make_tuple(
                          particles->get_pos_ptr(), particle_indices.addr())));

  // Create temporary arrays for all properties that need sorting
  DArray<float3> temp_vel(num);
  DArray<float> temp_mass(num);
  DArray<float> temp_merge(num);
  DArray<int> temp_merge_count(num);
  DArray<int> temp_remove(num);

  try {
    // Sort velocities
    thrust::gather(thrust::device, particle_indices.addr(),
                   particle_indices.addr() + num, particles->get_vel_ptr(),
                   temp_vel.addr());
    CUDA_CALL(cudaMemcpy(particles->get_vel_ptr(), temp_vel.addr(),
                         sizeof(float3) * num, cudaMemcpyDeviceToDevice));

    // Sort masses
    thrust::gather(thrust::device, particle_indices.addr(),
                   particle_indices.addr() + num, particles->get_mass_ptr(),
                   temp_mass.addr());
    CUDA_CALL(cudaMemcpy(particles->get_mass_ptr(), temp_mass.addr(),
                         sizeof(float) * num, cudaMemcpyDeviceToDevice));

    // Sort solver buffers
    _solver.resize(num);

    // Sort merge buffer
    thrust::gather(thrust::device, particle_indices.addr(),
                   particle_indices.addr() + num,
                   _solver.get_buffer_merge_ptr(), temp_merge.addr());
    CUDA_CALL(cudaMemcpy(_solver.get_buffer_merge_ptr(), temp_merge.addr(),
                         sizeof(float) * num, cudaMemcpyDeviceToDevice));

    // Sort merge count buffer
    thrust::gather(
        thrust::device, particle_indices.addr(), particle_indices.addr() + num,
        _solver.get_buffer_merge_count_ptr(), temp_merge_count.addr());
    CUDA_CALL(cudaMemcpy(_solver.get_buffer_merge_count_ptr(),
                         temp_merge_count.addr(), sizeof(int) * num,
                         cudaMemcpyDeviceToDevice));

    // Sort remove buffer
    thrust::gather(thrust::device, particle_indices.addr(),
                   particle_indices.addr() + num,
                   _solver.get_buffer_remove_ptr(), temp_remove.addr());
    CUDA_CALL(cudaMemcpy(_solver.get_buffer_remove_ptr(), temp_remove.addr(),
                         sizeof(int) * num, cudaMemcpyDeviceToDevice));

  } catch (const std::exception &e) {
    std::cerr << "Error in sorting particle properties: " << e.what()
              << std::endl;
    throw;
  }

  // Update particle_2_cell with sorted cell indices
  CUDA_CALL(cudaMemcpy(particles->get_particle_2_cell(), _buffer_int.addr(),
                       sizeof(int) * num, cudaMemcpyDeviceToDevice));

  // Initialize cell start array
  try {
    thrust::fill(
        thrust::device, cell_start.addr(),
        cell_start.addr() + _cell_size.x * _cell_size.y * _cell_size.z + 1, 0);
  } catch (const std::exception &e) {
    std::cerr << "Error filling cell_start: " << e.what() << std::endl;
    throw;
  }

  // Count particles per cell
  countingInCell_CUDA<<<(num - 1) / block_size + 1, block_size>>>(
      cell_start.addr(), particles->get_particle_2_cell(), num);

  // Calculate prefix sum for cell start array
  try {
    thrust::exclusive_scan(thrust::device, cell_start.addr(),
                           cell_start.addr() +
                               _cell_size.x * _cell_size.y * _cell_size.z + 1,
                           cell_start.addr());
  } catch (const std::exception &e) {
    std::cerr << "Error in exclusive_scan: " << e.what() << std::endl;
    throw;
  }

  cudaDeviceSynchronize();
}
void GranularSystem::neighbor_search_boundary(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {
  int num = particles->size();
  // std::cout << "Starting neighbor search for " << num << " particles"
  //           << std::endl;

  // Verify memory state
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

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

  try {
    thrust::sort_by_key(
        thrust::device, particles->get_particle_2_cell(),
        particles->get_particle_2_cell() + num,
        thrust::make_zip_iterator(thrust::make_tuple(
            particles->get_pos_ptr(), particles->get_is_animated_ptr())));
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

void GranularSystem::neighbor_search_upsampled(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {
  int num = particles->size();

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
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));
  CUDA_CALL(cudaEventRecord(start, 0));
  if (_is_move_boundary) {
    neighbor_search_boundary(_boundaries, _cell_start_boundary);
  }
  neighbor_search_granular(_particles, _cell_start_particle);
  neighbor_search_upsampled(_upsampled, _cell_start_upsampled);

  try {
    _solver.step(_particles, _boundaries, _cell_start_particle,
                 _cell_start_boundary, _space_size, _cell_size, _cell_length,
                 _dt, _g, _density);

    cudaDeviceSynchronize();

    _solver.upsampled_update(_particles, _boundaries, _upsampled,
                             _cell_start_upsampled, _cell_start_particle,
                             _cell_start_boundary, _cell_size, _space_size,
                             _cell_length, _density);

    if (_is_adaptive == 1) {
      set_surface_particles(_particles, _cell_start_particle);

      cudaDeviceSynchronize();

      find_distance_to_surface(_particles, _cell_start_particle);

      cudaDeviceSynchronize();

      _solver.adaptive_sampling(_particles, _boundaries, _cell_start_particle,
                                _cell_start_boundary, _max_mass, _cell_size,
                                _space_size, _cell_length, _density);
    }

  } catch (const char *s) {
    std::cout << s << "\n";
  } catch (...) {
    std::cout << "Unknown Exception at " << __FILE__ << ": line" << __LINE__
              << "\n";
  }

  float milliseconds;
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));

  frame_times.push_back(milliseconds); // Store the frame time
  std::cout << "Frame time : " << milliseconds << "\n";

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

__global__ void find_num_surface_neighbors(
    int *num_surface_neighbors, float3 *pos_granular, float *mass_granular,
    const int num, int *cell_start_granular, const int3 cell_size,
    const float cell_length, const float density, int *buffer_boundary) {

  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  __syncthreads();

  const float r_i = cbrtf((3 * mass_granular[i]) / (4 * PI_d * density));
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

      const float dis = length(pos_granular[j] - pos_granular[i]);
      const float r_j = cbrtf((3 * mass_granular[j]) / (4 * PI_d * density));

      if (dis > 2.0 * (r_i + r_j)) {
        j++;
        continue;
      }

      if (buffer_boundary[j] == 1) {
        num_surface_neighbors[i]++;
      }
      j++;
    }
  }
}

__global__ void find_cover_vector(float3 *buffer_cover_vector,
                                  float3 *pos_granular, float *mass_granular,
                                  const int num, int *cell_start_granular,
                                  const int3 cell_size, const float cell_length,
                                  const float density) {

  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  __syncthreads();

  float3 c_v = make_float3(0.0f, 0.0f, 0.0f);
  const float r_i = cbrtf((3 * mass_granular[i]) / (4 * PI_d * density));

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

      const float r_j = cbrtf((3 * mass_granular[j]) / (4 * PI_d * density));

      const float dis = length(pos_granular[i] - pos_granular[j]);

      if (dis > 2.0 * (r_i + r_j)) {
        j++;
        continue;
      }

      float3 b = pos_granular[i] - pos_granular[j];
      c_v += normalize(b);
      // NOTE: maybe set n_neighbors in the pbd loop
      j++;
    }
  }
  buffer_cover_vector[i] = c_v;
}

__global__ void find_surface(int *buffer_boundary, float3 *buffer_cover_vector,
                             float3 *pos_granular, float *mass_granular,
                             const int num, int *cell_start_granular,
                             const int3 cell_size, const float cell_length,
                             const float density) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (i >= num)
    return;
  __syncthreads();

  unsigned int n_neighbors = 0;
  float3 c_v = buffer_cover_vector[i];
  c_v = normalize(c_v);

  bool boundary_vote = false;
  const float r_i = cbrtf((3 * mass_granular[i]) / (4 * PI_d * density));

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
      float3 b = pos_granular[j] - pos_granular[i];
      const float r_j = cbrtf((3 * mass_granular[j]) / (4 * PI_d * density));
      const float dis = length(b);

      if (dis > 2.0 * (r_i + r_j)) {
        j++;
        continue;
      }

      b = normalize(b);

      if (acos(dot(b, c_v)) <= PI_3) {
        boundary_vote = true;
      }
      n_neighbors++;
      j++;
    }
  }

  if (n_neighbors <= 5 || !boundary_vote) {
    buffer_boundary[i] = 1;
  } else {
    buffer_boundary[i] = 0;
  }
}

void GranularSystem::set_surface_particles(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {
  const int num = particles->size();
  find_cover_vector<<<(num - 1) / block_size + 1, block_size>>>(
      _buffer_cover_vector.addr(), particles->get_pos_ptr(),
      particles->get_mass_ptr(), num, _cell_start_particle.addr(), _cell_size,
      _cell_length, _density);

  cudaDeviceSynchronize();

  find_surface<<<(num - 1) / block_size + 1, block_size>>>(
      _buffer_boundary.addr(), _buffer_cover_vector.addr(),
      particles->get_pos_ptr(), particles->get_mass_ptr(), num,
      _cell_start_particle.addr(), _cell_size, _cell_length, _density);

  CUDA_CALL(cudaMemcpy(particles->get_surface_ptr(), _buffer_boundary.addr(),
                       sizeof(int) * particles->size(),
                       cudaMemcpyDeviceToDevice));

  // find_num_surface_neighbors<<<(num - 1) / block_size + 1, block_size>>>(
  //     _buffer_num_surface_neighbors.addr(), particles->get_pos_ptr(),
  //     particles->get_mass_ptr(), num, _cell_start_particle.addr(),
  //     _cell_size, _cell_length, _density, _buffer_boundary.addr());
}

__global__ void
compute_distance_to_surface(const int num, const float3 *pos_particles,
                            const int *is_surface, const int *cell_start,
                            const int3 cell_size, const float cell_length,
                            float *min_distance2) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  __syncthreads();

  if (is_surface[i] == 1) {
    min_distance2[i] = 0.0f;
    return;
  }

  float best_dist2 = FLT_MAX;

  // Loop through 27 neighboring cells using the same pattern as other functions
  for (auto m = 0; m < 27; __syncthreads(), ++m) {
    const auto cellID = particlePos2cellIdx(
        make_int3(pos_particles[i] / cell_length) +
            make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
        cell_size);

    if (cellID == (cell_size.x * cell_size.y * cell_size.z))
      continue;

    // Get particles in this cell
    int j = cell_start[cellID];
    while (j < cell_start[cellID + 1]) {
      if (i == j) {
        j++;
        continue;
      }

      // Only consider surface particles
      if (is_surface[j] != 0) {
        float3 diff = pos_particles[i] - pos_particles[j];
        float dist2 = dot(diff, diff); // Using dot product from helper_math.h

        if (dist2 < best_dist2) {
          best_dist2 = dist2;
        }
      }
      j++;
    }
  }

  min_distance2[i] = best_dist2;
}

void GranularSystem::find_distance_to_surface(
    const std::shared_ptr<GranularParticles> &particles,
    DArray<int> &cell_start) {

  const int num = particles->size();
  compute_distance_to_surface<<<(num - 1) / block_size + 1, block_size>>>(
      num, particles->get_pos_ptr(), particles->get_surface_ptr(),
      _cell_start_particle.addr(), _cell_size, _cell_length,
      particles->get_surface_distance_ptr());
}
