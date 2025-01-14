#include "CUDAFunctions.cuh"
#include "Global.hpp"
#include "Solver.hpp"
#include "helper_math.h"
#include <algorithm>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <stdatomic.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <vector_types.h>

__device__ __constant__ double pi = 3.14159265358979323846;
#define EPSILON_m 1e-4f // Small threshold for comparison

void print_darray_int(const DArray<int> &_num_constraints) {
  // Step 1: Allocate host memory
  const unsigned int length = _num_constraints.length();
  std::vector<int> host_array(length);

  // Step 2: Copy data from device to host
  CUDA_CALL(cudaMemcpy(host_array.data(), _num_constraints.addr(),
                       sizeof(int) * length, cudaMemcpyDeviceToHost));

  // Step 3: Print the data
  for (size_t i = 0; i < length; ++i) {
    std::cout << "Constraint[" << i << "] = " << host_array[i] << std::endl;
  }
}

void print_mass(const DArray<float> &mass) {
  // Step 1: Allocate host memory
  const unsigned int length = mass.length();
  std::vector<float> host_array(length);

  // Step 2: Copy data from device to host
  CUDA_CALL(cudaMemcpy(host_array.data(), mass.addr(), sizeof(float) * length,
                       cudaMemcpyDeviceToHost));

  // Step 3: Print the data
  for (size_t i = 0; i < length; ++i) {
    std::cout << "Mass[" << i << "] = " << host_array[i] << std::endl;
  }
}

void print_positions(const DArray<float3> &positions) {
  const unsigned int length = positions.length();
  std::vector<float3> host_array(length);

  CUDA_CALL(cudaMemcpy(host_array.data(), positions.addr(),
                       sizeof(float3) * length, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < length; ++i) {
    std::cout << "Position[" << i << "] = (" << host_array[i].x << ", "
              << host_array[i].y << ", " << host_array[i].z << ")" << std::endl;
  }
}

void Solver::step(std::shared_ptr<GranularParticles> &particles,
                  const std::shared_ptr<GranularParticles> &boundary,
                  const DArray<int> &cell_start_granular,
                  const DArray<int> &cell_start_boundary, float3 space_size,
                  int3 cell_size, float cell_length, float dt, float3 G,
                  const float density) {

  _buffer_int.resize(particles->size());
  // apply forces
  // update velocity
  add_external_force(particles, dt, G);
  update_particle_positions(particles, dt);

  // update_neighborhood(particles);
  // project constraints
  project(particles, boundary, cell_start_granular, cell_start_boundary,
          cell_size, space_size, cell_length, 5, density);

  // TODO: resize remaning stuff

  final_update(particles, dt);
}

// WARNING: Seems to cause issues with incorrect neighbors
void Solver::update_neighborhood(
    const std::shared_ptr<GranularParticles> &particles) {

  const int num = particles->size();
  CUDA_CALL(cudaMemcpy(_buffer_int.addr(), particles->get_particle_2_cell(),
                       sizeof(int) * num, cudaMemcpyDeviceToDevice));
  // NOTE: might need to fix the value
  thrust::sort_by_key(thrust::device, _buffer_int.addr(),
                      _buffer_int.addr() + num, particles->get_pos().addr());

  return;
}

void Solver::add_external_force(std::shared_ptr<GranularParticles> &particles,
                                float dt, float3 G) {
  const auto dv = dt * G;
  thrust::transform(thrust::device, particles->get_vel_ptr(),
                    particles->get_vel_ptr() + particles->size(),
                    particles->get_vel_ptr(), ThrustHelper::plus<float3>(dv));
}

struct predict_position_functor {
  float dt;

  predict_position_functor(float _dt) : dt(_dt) {}

  __host__ __device__ float3
  operator()(const thrust::tuple<float3, float3> &t) const {
    const float3 &pos = thrust::get<0>(t);
    const float3 &vel = thrust::get<1>(t);
    return make_float3(pos.x + dt * vel.x, pos.y + dt * vel.y,
                       pos.z + dt * vel.z);
  }
};

void Solver::update_particle_positions(
    std::shared_ptr<GranularParticles> &particles, float dt) {
  // Assuming particles->get_pos_ptr() returns a pointer to the first element of
  // the position buffer and particles->get_vel_ptr() returns a pointer to the
  // first element of the velocity buffer

  // Create zip iterator for positions and velocities
  // We use a zip iterator because we need to loop through postions and
  // velocties together
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr(), particles->get_vel_ptr()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr() + particles->size(),
                         particles->get_vel_ptr() + particles->size()));

  // Update positions by applying the 'update_position_functor' across the range
  thrust::transform(
      thrust::device, begin, end, _pos_t.addr(),
      // particles->get_pos_ptr(), // Output to the positions buffer
      predict_position_functor(dt));
}

struct final_velocity_functor {
  float dt;

  final_velocity_functor(float _dt) : dt(_dt) {}

  __host__ __device__ float3
  operator()(const thrust::tuple<float3, float3> &t) const {
    const float dt_inv = 1 / dt;
    const float3 &pos = thrust::get<0>(t);
    const float3 &pos_t = thrust::get<1>(t);

    return make_float3(dt_inv * (pos_t.x - pos.x), dt_inv * (pos_t.y - pos.y),
                       dt_inv * (pos_t.z - pos.z));
  }
};

void Solver::final_update(std::shared_ptr<GranularParticles> &particles,
                          float dt) {

  // update velocity
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr(), _pos_t.addr()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr() + particles->size(),
                         _pos_t.addr() + particles->size()));

  thrust::transform(
      thrust::device, begin, end, particles->get_vel_ptr(),
      // particles->get_pos_ptr(), // Output to the positions buffer
      final_velocity_functor(dt));

  // update position
  CUDA_CALL(cudaMemcpy(particles->get_pos_ptr(), _pos_t.addr(),
                       sizeof(float3) * particles->size(),
                       cudaMemcpyDeviceToDevice));
}

__device__ void boundary_constraint(float3 &del_p, int &n, int i,
                                    const float3 pos_p, float3 *pos_b, float *m,
                                    int j, const int cell_end,
                                    const int density) {
  while (j < cell_end) {
    const float dis = norm3df(pos_p.x - pos_b[j].x, pos_p.y - pos_b[j].y,
                              pos_p.z - pos_b[j].z);

    const float r_i = cbrtf((3 * m[i]) / (4 * pi * density));
    const float mag = dis - (0.01 + r_i);
    const float3 p_12 = pos_p - pos_b[j];
    if (mag < 0.0) {
      del_p -= (mag / dis) * p_12;
      n++;
    }
    ++j;
  }
  return;
}

__device__ void particles_constraint(float3 &del_p, int &n, int i,
                                     float3 *pos_p, float *m, int j,
                                     const int cell_end, const int density,
                                     int *n_constraints, float3 *delta_pos) {
  while (j < cell_end) {
    if (i != j) {
      const float3 p_12 = pos_p[i] - pos_p[j];
      const float inv_m_i = 1 / m[i];
      const float inv_m_j = 1 / m[j];
      const float inv_m_sum = 1.0 / (inv_m_i + inv_m_j);
      const float r_i = cbrtf((3 * m[i]) / (4 * pi * density));
      const float r_j = cbrtf((3 * m[j]) / (4 * pi * density));
      if (i == 0) {

        // printf("m_1 = %f, m_2 = %f\n", inv_m_i, inv_m_j);
        // printf("r_1 = %f, r_2 = %f\n", r_i, r_j);
        // printf("mass_sum = %f\n", inv_m_sum);
      }
      const float dis =
          norm3df(pos_p[i].x - pos_p[j].x, pos_p[i].y - pos_p[j].y,
                  pos_p[i].z - pos_p[j].z);
      const float mag = dis - (r_i + r_j);

      // TODO: add mass scaling
      if (mag < 0.0) {
        del_p -= inv_m_sum * inv_m_i * (mag / dis) * p_12;
        delta_pos[j] += inv_m_sum * inv_m_j * (mag / dis) * p_12;

        n++;
        n_constraints[j]++;
      }
    }
    ++j;
  }
  return;
}

__global__ void compute_delta_pos(float3 *delta_pos, int *n,
                                  float3 *pos_granular, float3 *pos_boundary,
                                  float *mass_granular, const int num,
                                  int *cell_start_granular,
                                  int *cell_start_boundary,
                                  const int3 cell_size, const float cell_length,
                                  const int density) {

  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  // out of bounds
  if (i >= num)
    return;
  float3 dp = make_float3(0.0f);

  __syncthreads();

#pragma unroll
  // Loop through the 27 neighboring cells
  for (auto m = 0; m < 27; __syncthreads(), ++m) {
    const auto cellID = particlePos2cellIdx(
        make_int3(pos_granular[i] / cell_length) +
            make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
        cell_size);
    if (cellID == (cell_size.x * cell_size.y * cell_size.z))
      continue;
    // contributeDeltaPos_fluid(dp, i, pos_fluid, lambda, massFluid,
    //                          cellStartFluid[cellID], cellStartFluid[cellID +
    //                          1], radius);
    boundary_constraint(dp, n[i], i, pos_granular[i], pos_boundary,
                        mass_granular, cell_start_boundary[cellID],
                        cell_start_boundary[cellID + 1], density);

    particles_constraint(
        dp, n[i], i, pos_granular, mass_granular, cell_start_granular[cellID],
        cell_start_granular[cellID + 1], density, n, delta_pos);
  }

  delta_pos[i] = dp;
  return;
}

struct change_position_functor {

  change_position_functor() {}

  __host__ __device__ float3
  operator()(const thrust::tuple<float3, int, float3> &t) const {
    const float3 &del_pos = thrust::get<0>(t);
    const int &n = max(1, thrust::get<1>(t));
    const float3 &pos_t = thrust::get<2>(t);
    return pos_t + del_pos / n;
  }
};

void Solver::project(std::shared_ptr<GranularParticles> &particles,
                     const std::shared_ptr<GranularParticles> &boundaries,
                     const DArray<int> &cell_start_granular,
                     const DArray<int> &cell_start_boundary, int3 cell_size,
                     float3 space_size, float cell_length, int max_iter,
                     const float density) {

  int iter = 0;
  int stab_iter = 0;
  const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
  const int num = particles->size();

  while (stab_iter < 2) {

    // reset change in position and number of elements
    thrust::device_ptr<float3> thrust_buffer_float_3 =
        thrust::device_pointer_cast(_buffer_float3.addr());
    thrust::fill(thrust_buffer_float_3, thrust_buffer_float_3 + num, zero);

    thrust::device_ptr<int> thrust_num_constraints =
        thrust::device_pointer_cast(_num_constraints.addr());

    thrust::fill(thrust_num_constraints, thrust_num_constraints + num, 0);

    // calculate change in positon
    compute_delta_pos<<<(num - 1) / block_size + 1, block_size>>>(

        _buffer_float3.addr(), _num_constraints.addr(), _pos_t.addr(),
        boundaries->get_pos_ptr(), particles->get_mass_ptr(), num,
        cell_start_granular.addr(), cell_start_boundary.addr(), cell_size,
        cell_length, density);

    // update the position
    auto begin_p = thrust::make_zip_iterator(
        thrust::make_tuple(_buffer_float3.addr(), _num_constraints.addr(),
                           particles->get_pos_ptr()));

    auto end_p = thrust::make_zip_iterator(thrust::make_tuple(
        _buffer_float3.addr() + num, _num_constraints.addr() + num,
        particles->get_pos_ptr() + num));

    thrust::transform(
        thrust::device, begin_p, end_p, particles->get_pos_ptr(),
        // particles->get_pos_ptr(), // Output to the positions buffer
        change_position_functor());

    // update chage in position
    auto begin_del_p = thrust::make_zip_iterator(thrust::make_tuple(
        _buffer_float3.addr(), _num_constraints.addr(), _pos_t.addr()));

    auto end_del_p = thrust::make_zip_iterator(
        thrust::make_tuple(_buffer_float3.addr() + num,
                           _num_constraints.addr() + num, _pos_t.addr() + num));

    thrust::transform(
        thrust::device, begin_del_p, end_del_p, _pos_t.addr(),
        // particles->get_pos_ptr(), // Output to the positions buffer
        change_position_functor());

    stab_iter++;
  }

  while (iter < max_iter) {

    // reset delta p and num constraints
    thrust::device_ptr<float3> thrust_buffer_float_3 =
        thrust::device_pointer_cast(_buffer_float3.addr());
    thrust::fill(thrust_buffer_float_3, thrust_buffer_float_3 + num, zero);

    thrust::device_ptr<int> thrust_num_constraints =
        thrust::device_pointer_cast(_num_constraints.addr());

    thrust::fill(thrust_num_constraints, thrust_num_constraints + num, 0);

    // print_darray_int(_num_constraints);

    compute_delta_pos<<<(num - 1) / block_size + 1, block_size>>>(

        _buffer_float3.addr(), _num_constraints.addr(), _pos_t.addr(),
        boundaries->get_pos_ptr(), particles->get_mass_ptr(), num,
        cell_start_granular.addr(), cell_start_boundary.addr(), cell_size,
        cell_length, density);
    //
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        _buffer_float3.addr(), _num_constraints.addr(), _pos_t.addr()));

    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(_buffer_float3.addr() + num,
                           _num_constraints.addr() + num, _pos_t.addr() + num));

    thrust::transform(
        thrust::device, begin, end, _pos_t.addr(),
        // particles->get_pos_ptr(), // Output to the positions buffer
        change_position_functor());

    iter++;
  }
}

__device__ void viable_merge(const int i, int j, const int cell_end,
                             const float *m, const int *surface,
                             const int *remove, const float max_mass, int &n,
                             int *n_indices) {
  while (j < cell_end) {
    if (i != j) {
      if (remove[j] == 1) {
        n++;
      }
    }
    ++j;
  }
}

// __global__ void merge_mark_gpu(const int num, float3 *pos_granular,
//                                float *mass_granular, int *surface, int
//                                *remove, float *merge, int
//                                *cell_start_granular, float max_mass, const
//                                int3 cell_size, const float cell_length) {
//
//   const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//   // Bounds check
//   if (i >= num) {
//     return;
//   }
//
//   // Validate input data
//   if (!pos_granular || !mass_granular || !surface || !remove || !merge ||
//       !cell_start_granular) {
//     printf("Kernel error: null pointer detected\n");
//     return;
//   }
//
//   // Only merge if the particle is not a surface particle, not marked to be
//   // removed and not marked to be merged
//   if (surface[i] == 0 && atomicOr(&remove[i], 0) == 0) {
//     float closest_dis = 1000.0f;
//     int closest_index = -1;
//
// #pragma unroll
//     // Loop through the 27 neighboring cells
//     for (auto m = 0; m < 27; __syncthreads(), ++m) {
//       const auto cellID = particlePos2cellIdx(
//           make_int3(pos_granular[i] / cell_length) +
//               make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
//           cell_size);
//
//       if (cellID >= cell_size.x * cell_size.y * cell_size.z) {
//         printf("Invalid cell ID computed for particle %d\n", i);
//         return;
//       }
//
//       if (cellID == (cell_size.x * cell_size.y * cell_size.z))
//         continue;
//
//       int j = cell_start_granular[cellID];
//       while (j < cell_start_granular[cellID + 1] && j < num) {
//         // Add debug print when we encounter a merging particle
//         if (atomicOr(&remove[j], 0) != 0) {
//           printf("Particle %d encountered particle %d which has remove flag =
//           "
//                  "%d and mass = %f\n",
//                  i, j, remove[j], mass_granular[j]);
//         }
//         // if j is marked for removal or marked to be merged or a surface
//         // particle
//         if (j <= i || atomicOr(&remove[j], 0) != 0 || surface[j] == 1) {
//           j++;
//           continue;
//         }
//
//         const bool mass_check = mass_granular[j] <= max_mass -
//         mass_granular[i];
//
//         if (!mass_check) {
//           atomicAdd(&j, 1);
//           continue;
//         }
//
//         const float3 p_i = pos_granular[i];
//         const float3 p_j = pos_granular[j];
//
//         const float dis =
//             norm3df((p_i.x - p_j.x), (p_i.y - p_j.y), (p_i.z - p_j.z));
//
//         if (dis < closest_dis) {
//           closest_dis = dis;
//           atomicExch(&closest_index, j);
//         }
//         atomicAdd(&j, 1);
//       }
//     }
//
//     // we found a viable candidate
//     if (closest_index != -1) {
//       // Try to mark both particles atomically
//       if (atomicCAS(&remove[i], 0, -1) == 0) { // First try to mark i
//         if (atomicCAS(&remove[closest_index], 0, -1) ==
//             0) { // Then try to mark closest_index
//           // Both particles were successfully marked
//           atomicExch(&merge[closest_index], mass_granular[i]);
//           atomicExch(&merge[i], -mass_granular[i]);
//           printf("Setting up merge: particle %d (mass %.3f) -> particle %d "
//                  "(mass %.3f), delta: %f and %f\n",
//                  i, mass_granular[i], closest_index,
//                  mass_granular[closest_index], mass_granular[i],
//                  -mass_granular[i]);
//         } else {
//           // Failed to mark closest_index, revert i's marking
//           atomicExch(&remove[i], 0);
//         }
//       }
//     }
//
//     // Try to atomically acquire the merge lock
//     // if (atomicCAS(&remove[closest_index], 0, -1) == 0) {
//     //   atomicExch(&remove[i], 1);
//     //   // NOTE: ??
//     //   // atomicExch(&merge[i], closest_index);
//     //   merge[closest_index] = mass_granular[i];
//     // }
//   }
//   return;
// }

__global__ void merge_mark_gpu(const int num, float3 *pos_granular,
                               float *mass_granular, int *surface, int *remove,
                               float *merge, int *cell_start_granular,
                               float max_mass, const int3 cell_size,
                               const float cell_length) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  // Only non-surface particles that aren't already marked
  if (surface[i] != 0 || atomicOr(&remove[i], 0) != 0)
    return;

  float3 pos_i = pos_granular[i];
  float mass_i = mass_granular[i];
  float closest_dis = 1000.0f;
  int closest_index = -1;

#pragma unroll
  for (auto m = 0; m < 27; __syncthreads(), ++m) {
    const auto cellID = particlePos2cellIdx(
        make_int3(pos_i / cell_length) +
            make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
        cell_size);

    if (cellID == (cell_size.x * cell_size.y * cell_size.z))
      continue;

    int j = cell_start_granular[cellID];
    while (j < cell_start_granular[cellID + 1] && j < num) {
      if (j <= i || atomicOr(&remove[j], 0) != 0 || surface[j] != 0) {
        j++;
        continue;
      }

      // Only merge if combined mass is valid
      float combined_mass = mass_i + mass_granular[j];
      if (combined_mass > max_mass) {
        j++;
        continue;
      }

      float3 p_j = pos_granular[j];
      float dis = norm3df(pos_i.x - p_j.x, pos_i.y - p_j.y, pos_i.z - p_j.z);

      if (dis < closest_dis) {
        closest_dis = dis;
        closest_index = j;
      }
      j++;
    }
  }

  // Only proceed if we found a merge candidate
  if (closest_index != -1) {
    if (atomicCAS(&remove[i], 0, -1) == 0) {
      if (atomicCAS(&remove[closest_index], 0, -1) == 0) {
        atomicExch(&merge[closest_index], mass_i);
        atomicExch(&merge[i], -mass_i);
        printf("Merge set up: particle %d (mass %.3f) -> particle %d (mass "
               "%.3f)\n",
               i, mass_i, closest_index, mass_granular[closest_index]);
      } else {
        atomicExch(&remove[i], 0);
      }
    }
  }
}
__device__ bool isAlmostZero(float x) {
  return fabsf(x) < EPSILON_m; // fabsf for single precision
}

__global__ void merge_count_gpu(const int num, float *mass_del,
                                float *mass_granular, int *merge_count,
                                int *remove, const int blend_factor) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (remove[i] == -1) {
    int old_count = atomicAdd(&merge_count[i], 1);
    float delta = mass_del[i] / blend_factor;
    atomicAdd(&mass_granular[i], delta);

    // Debug
    printf("Particle %d: step %d/%d, mass %.3f, delta %.3f\n", i, old_count + 1,
           blend_factor, mass_granular[i], delta);

    if (old_count + 1 == blend_factor) {
      if (mass_del[i] < 0) {
        // Only mark for removal if this particle is giving mass
        atomicExch(&remove[i], 1);
        printf("Particle %d marked for removal (final mass %.3f)\n", i,
               mass_granular[i]);
      } else {
        atomicExch(&remove[i], 0);
        printf("Particle %d merge complete (final mass %.3f)\n", i,
               mass_granular[i]);
      }
      atomicExch(&merge_count[i], 0);
      atomicExch(&mass_del[i], 0.0f);
    }
  }
}

__global__ void split_gpu(const int num, float3 *pos_granular,
                          float *mass_granular, float3 *vel_granular,
                          int *surface, int *remove, float *merge,
                          int *cell_start_granular, float min_mass,
                          const int3 cell_size, float *split_mass,
                          float3 *split_pos, float3 *split_vel,
                          int &split_count, float density) {

  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  // Merging/removing particles cannot split
  if (i >= num || atomicOr(&remove[i], 0) != 0 || surface[i] != 1)
    return;

  if (mass_granular[i] >= 2 * min_mass) {

    const float new_mass = mass_granular[i] / 2;
    const float3 new_vel = vel_granular[i];

    // TODO: make the position selection random

    const float r_i = cbrtf((3 * mass_granular[i]) / (4 * pi * density));

    const float3 new_pos_1 = make_float3(pos_granular[i].x + r_i,
                                         pos_granular[i].y, pos_granular[i].z);
    const float3 new_pos_2 = make_float3(pos_granular[i].x - r_i,
                                         pos_granular[i].y, pos_granular[i].z);

    mass_granular[i] = new_mass;
    atomicExch(&split_mass[split_count], new_mass);

    vel_granular[i] = new_vel;
    split_vel[split_count] = new_vel;

    pos_granular[i] = new_pos_1;
    split_pos[split_count] = new_pos_2;

    atomicAdd(&split_count, 1);
    // split
  }
  return;
}

struct merge_functor {

  merge_functor() {}

  __host__ __device__ float3
  operator()(const thrust::tuple<float, float> &t) const {
    const float &mass = thrust::get<0>(t);
  }
};

// Host function
void Solver::adaptive_sampling(std::shared_ptr<GranularParticles> &particles,
                               const DArray<int> &cell_start_granular,
                               const float max_mass, int3 cell_size,
                               float3 space_size, float cell_length,
                               const float density) {
  const int num = particles->size();
  if (num == 0)
    return;

  cudaDeviceSynchronize();

  try {
    // old mass

    auto m_t = thrust::device_pointer_cast(particles->get_mass_ptr());

    const float old_mass =
        thrust::reduce(m_t, m_t + num, 0, thrust::plus<float>());

    // Launch kernel with error checking
    cudaError_t err = cudaSuccess;

    // Run merge kernel
    merge_mark_gpu<<<(num + block_size - 1) / block_size, block_size>>>(
        num, particles->get_pos_ptr(), particles->get_mass_ptr(),
        particles->get_surface_ptr(), _buffer_remove.addr(),
        _buffer_merge.addr(), cell_start_granular.addr(), max_mass, cell_size,
        cell_length);

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
                << std::endl;
      throw std::runtime_error("Kernel launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
                << std::endl;
      throw std::runtime_error("Kernel execution failed");
    }

    // Print info about particles being removed
    std::vector<int> host_remove(num);
    std::vector<float> host_mass(num);
    CUDA_CALL(cudaMemcpy(host_remove.data(), _buffer_remove.addr(),
                         sizeof(int) * num, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_mass.data(), particles->get_mass_ptr(),
                         sizeof(float) * num, cudaMemcpyDeviceToHost));

    int removal_count = 0;
    std::cout << "Particles marked for removal (mass):\n";
    for (int i = 0; i < num; i++) {
      if (host_remove[i] == 1) {
        removal_count++;
        std::cout << "Particle " << i << ": mass = " << host_mass[i]
                  << ", remove flag = " << host_remove[i]
                  << ", merge mass delta = ";

        // Get merge mass delta for this particle
        float merge_delta;
        CUDA_CALL(cudaMemcpy(&merge_delta, &_buffer_merge.addr()[i],
                             sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << merge_delta << "\n";
      }
    }
    std::cout << "Total particles marked for removal: " << removal_count
              << "\n";

    // Run split kernel
    int n_split = 0;
    DArray<float> split_mass(1000);
    DArray<float3> split_pos(1000);
    DArray<float3> split_vel(1000);

    split_gpu<<<(num + block_size - 1) / block_size, block_size>>>(
        num, particles->get_pos_ptr(), particles->get_mass_ptr(),
        particles->get_vel_ptr(), particles->get_surface_ptr(),
        _buffer_remove.addr(), _buffer_merge.addr(), cell_start_granular.addr(),
        max_mass, cell_size, split_mass.addr(), split_pos.addr(),
        split_vel.addr(), n_split, density);

    cudaDeviceSynchronize();

    split_mass.resize(n_split);
    split_pos.resize(n_split);
    split_vel.resize(n_split);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("Merge kernel error: ") +
                               cudaGetErrorString(err));
    }

    // TODO: Velocity update
    // thrust::transform(thrust::device, particles->get_mass_ptr(),
    //                   particles->get_mass_ptr() + num,
    //                   _buffer_merge.addr(), particles->get_mass_ptr(),
    //                   thrust::plus<float>());
    //

    // gradual merging
    merge_count_gpu<<<(num + block_size - 1) / block_size, block_size>>>(
        num, _buffer_merge.addr(), particles->get_mass_ptr(),
        _buffer_merge_count.addr(), _buffer_remove.addr(), _blend_factor);

    CUDA_CALL(cudaDeviceSynchronize());

    // Print final state before removal
    std::cout << "\nFinal state before removal:\n";
    CUDA_CALL(cudaMemcpy(host_mass.data(), particles->get_mass_ptr(),
                         sizeof(float) * num, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_remove.data(), _buffer_remove.addr(),
                         sizeof(int) * num, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num; i++) {
      if (host_remove[i] == 1) {
        std::cout << "Particle " << i << ": final mass = " << host_mass[i]
                  << "\n";
      }
    }

    // remove elements
    particles->remove_elements(_buffer_remove);
    _buffer_merge_count.compact(_buffer_remove);
    _buffer_merge.compact(_buffer_remove);
    _buffer_remove.compact(_buffer_remove);

    CUDA_CALL(cudaDeviceSynchronize());

    // add elements
    particles->add_elements(split_mass, split_pos, split_vel, n_split);
    _buffer_merge_count.resize(particles->size());
    _buffer_merge.resize(particles->size());
    _buffer_remove.resize(particles->size());

    cudaDeviceSynchronize();

    // change in mass

    // Print total mass after
    auto m_t_n = thrust::device_pointer_cast(particles->get_mass_ptr());
    const float new_mass = thrust::reduce(m_t_n, m_t_n + particles->size(), 0,
                                          thrust::plus<float>());
    std::cout << "Total mass after: " << new_mass << "\n";
    if ((new_mass - old_mass) != 0) {
      std::cout << "Change in mass " << new_mass - old_mass << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error in adaptive_sampling: " << e.what() << std::endl;
    return;
  }
}
