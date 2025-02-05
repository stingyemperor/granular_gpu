#include "CUDAFunctions.cuh"
#include "Global.hpp"
#include "GranularParticles.hpp"
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
#include <unistd.h>
#include <vector_types.h>

__device__ __constant__ double pi = 3.14159265358979323846;
__device__ __constant__ float r_9 = 493.8271605;
__device__ __constant__ float r_9_b = 1111.1111111;
// __device__ __constant__ float r_9 = 277.7777778;

#define EPSILON_m 1e-4f // Small threshold for comparison

int t_merge_iter = 0;

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
  for (size_t i = -1; i < length; ++i) {
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
  apply_mass_scaling(particles);

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

struct mass_scaling_functor {
  // float _min_mass;
  // float _max_mass;
  // float _max_height;
  // float _min_height;

  // mass_scaling_functor(float min_mass, float max_mass, float max_height,
  //                      float min_height)
  //     : _min_mass(min_mass), _max_mass(max_mass), _min_height(min_height),
  //       _max_height(max_height) {}
  mass_scaling_functor() {}
  __host__ __device__ float
  operator()(const thrust::tuple<float, float3> &t) const {
    const float &mass = thrust::get<0>(t);
    const float3 &pos = thrust::get<1>(t);

    return mass * exp(-pos.y);
  }
};

void Solver::apply_mass_scaling(std::shared_ptr<GranularParticles> &particles) {
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_mass_ptr(), particles->get_pos_ptr()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_mass_ptr() + particles->size(),
                         particles->get_pos_ptr() + particles->size()));

  thrust::transform(thrust::device, begin, end,
                    particles->get_scaled_mass_ptr(), mass_scaling_functor());
}

struct final_velocity_functor {
  float dt;
  float min_speed;   // Speed threshold below which damping is minimal
  float max_speed;   // Speed threshold for maximum damping
  float min_damping; // Minimum damping factor (for low velocities)
  float max_damping; // Maximum damping factor (for high velocities)

  final_velocity_functor(float _dt)
      : dt(_dt),
        min_speed(1.0f),    // Adjust these thresholds based on your simulation
        max_speed(10.0f),   // Adjust these thresholds based on your simulation
        min_damping(0.99f), // Almost no damping for slow particles
        max_damping(0.7f)   // Stronger damping for fast particles
  {}

  __host__ __device__ float3
  operator()(const thrust::tuple<float3, float3> &t) const {
    const float dt_inv = 1.0f / dt;
    const float3 &pos = thrust::get<0>(t);
    const float3 &pos_t = thrust::get<1>(t);

    // Calculate raw velocity
    float3 vel =
        make_float3(dt_inv * (pos_t.x - pos.x), dt_inv * (pos_t.y - pos.y),
                    dt_inv * (pos_t.z - pos.z));

    // Calculate speed
    float speed = length(vel);

    if (speed > min_speed) {
      // Calculate damping factor based on speed
      float t =
          clamp((speed - min_speed) / (max_speed - min_speed), 0.0f, 1.0f);

      // Smooth interpolation between min and max damping
      float damping = min_damping + (max_damping - min_damping) * t;

      // Apply non-linear damping
      float damping_factor = damping + (1.0f - damping) * expf(-speed * 0.1f);

      // Apply damping
      vel *= damping_factor;
    }

    return vel;
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

    const float r_i = max(cbrtf((3 * m[i]) / (4 * pi * density)), 0.01f);
    const float mag = dis - (0.01 + r_i);
    const float3 p_12 = pos_p - pos_b[j];

    if (mag < 0.0) {
      const float3 del_p_i = (mag / dis) * p_12;
      const float3 del_p_i_perp =
          del_p_i - dot(del_p_i, p_12) * p_12 / (dis * dis);
      const float del_p_i_norm = norm3df(del_p_i.x, del_p_i.y, del_p_i.z);

      const float del_p_i_perp_norm =
          norm3df(del_p_i_perp.x, del_p_i_perp.y, del_p_i_perp.z);

      float min_fric = min((0.01 + r_i) * 0.5 / del_p_i_perp_norm, 1.0f);

      if (del_p_i_perp_norm < (r_i + 0.01) * 0.5) {
        del_p -= del_p_i;
      } else {
        del_p -= del_p_i * min_fric;
      }
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
      const float dis =
          norm3df(pos_p[i].x - pos_p[j].x, pos_p[i].y - pos_p[j].y,
                  pos_p[i].z - pos_p[j].z);
      const float mag = (r_i + r_j) - dis;

      // TODO: add mass scaling
      if (mag >= 0.0) {
        // del_p -= inv_m_sum * inv_m_i * (mag / dis) * p_12;
        // delta_pos[j] += inv_m_sum * inv_m_j * (mag / dis) * p_12;

        const float3 del_p_i = -inv_m_sum * inv_m_i * (mag / dis) * p_12;
        const float3 del_p_j = inv_m_sum * inv_m_j * (mag / dis) * p_12;

        const float3 del_p_ij = del_p_i - del_p_j;
        const float3 del_p_ij_perp =
            del_p_ij - dot(del_p_ij, p_12) * p_12 / (dis * dis);

        const float del_p_ij_perp_norm =
            norm3df(del_p_ij_perp.x, del_p_ij_perp.y, del_p_ij_perp.z);

        const float min_fric =
            min((r_i + r_j) * 0.5 / del_p_ij_perp_norm, 1.0f);

        if (del_p_ij_perp_norm < (r_i + r_j) * 0.5) {
          del_p -= inv_m_sum * inv_m_i * del_p_ij;
          delta_pos[j] += inv_m_sum * inv_m_i * del_p_ij;
        } else {
          del_p -= inv_m_sum * inv_m_i * del_p_ij * min_fric;
          delta_pos[j] += inv_m_sum * inv_m_i * del_p_ij * min_fric;
        }

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

  while (stab_iter < 3) {

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
        boundaries->get_pos_ptr(), particles->get_scaled_mass_ptr(), num,
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

__global__ void merge_mark_gpu(const int num, float3 *pos_granular,
                               float *mass_granular, float3 *vel_granular,
                               int *surface, int *num_surface_neighbors,
                               int *remove, float *merge,
                               float3 *merge_velocity, int *cell_start_granular,
                               float max_mass, const int3 cell_size,
                               const float cell_length) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  // Only non-surface particles that aren't already marked
  // if (surface[i] != 0 || atomicOr(&remove[i], 0) != 0 ||
  //     num_surface_neighbors[i] > 3)
  //   return;

  float mass_i = mass_granular[i];
  if (surface[i] != 0 || atomicOr(&remove[i], 0) != 0 || mass_i < 1.0f ||
      mass_i >= max_mass) {
    return;
  }

  float3 pos_i = pos_granular[i];
  float3 vel_i = vel_granular[i];
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
      // if (j <= i || atomicOr(&remove[j], 0) != 0 || surface[j] != 0 ||
      //     num_surface_neighbors[j] > 3) {
      //
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
      float dis = length(pos_i - p_j);
      // float dis = norm3df(pos_i.x - p_j.x, pos_i.y - p_j.y, pos_i.z - p_j.z);

      if (dis < 0.035) {
        if (dis < closest_dis) {
          closest_dis = dis;
          closest_index = j;
        }
      }
      j++;
    }
  }

  // Only proceed if we found a merge candidate
  if (closest_index != -1) {
    if (atomicCAS(&remove[i], 0, -1) == 0 && surface[i] == 0) {
      if (atomicCAS(&remove[closest_index], 0, -1) == 0 &&
          surface[closest_index] == 0) {
        atomicExch(&merge[closest_index], mass_i);
        atomicExch(&merge[i], -mass_i);

        const float m_t = 1 / (mass_i + mass_granular[closest_index]);
        const float3 vel_t =
            m_t * (mass_i * vel_i + mass_granular[closest_index] +
                   vel_granular[closest_index]);

        atomicExch(&merge_velocity[closest_index].x, vel_t.x);
        atomicExch(&merge_velocity[closest_index].y, vel_t.y);
        atomicExch(&merge_velocity[closest_index].z, vel_t.z);

        atomicExch(&merge_velocity[i].x, -vel_t.x);
        atomicExch(&merge_velocity[i].y, -vel_t.y);
        atomicExch(&merge_velocity[i].z, -vel_t.z);

        // printf("Merge set up: particle %d (mass %.3f) -> particle %d (mass "
        //        "%.3f)\n",
        //        i, mass_i, closest_index, mass_granular[closest_index]);
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
                                float *mass_granular, float3 *vel_granular,
                                int *merge_count, int *remove, float3 *vel_del,
                                const int blend_factor) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (remove[i] == -1) {
    int old_count = atomicAdd(&merge_count[i], 1);
    const float delta = mass_del[i] / blend_factor;
    const float3 delta_vel = vel_del[i] / blend_factor;
    mass_granular[i] += delta;
    // vel_granular[i] += delta_vel;

    // Debug
    // printf("Particle %d: step %d/%d, mass %.3f, delta %.3f\n", i, old_count +
    // 1,
    //        blend_factor, mass_granular[i], delta);

    if (old_count + 1 == blend_factor) {
      if (mass_del[i] < 0) {
        // Only mark for removal if this particle is giving mass
        atomicExch(&remove[i], 1);
        // printf("Particle %d marked for removal (final mass %.3f)\n", i,
        //        mass_granular[i]);
      } else {
        atomicExch(&remove[i], 0);
        // printf("Particle %d merge complete (final mass %.3f)\n", i,
        // mass_granular[i]);
      }
      atomicExch(&merge_count[i], 0);
      atomicExch(&mass_del[i], 0.0f);
      // vel_granular[i].x = 0;
      // vel_granular[i].y = 0;
      // vel_granular[i].z = 0;
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

  // Store initial state and particle IDs
  std::vector<float> initial_masses(num);
  std::vector<float3> initial_positions(num);
  std::vector<float3> initial_velocities(num);
  std::vector<int> remove_flags(num);
  std::vector<float> merge_values(num);

  // Copy initial data to host
  CUDA_CALL(cudaMemcpy(initial_masses.data(), particles->get_mass_ptr(),
                       sizeof(float) * num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(initial_positions.data(), particles->get_pos_ptr(),
                       sizeof(float3) * num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(initial_velocities.data(), particles->get_vel_ptr(),
                       sizeof(float3) * num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(remove_flags.data(), _buffer_remove.addr(),
                       sizeof(int) * num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(merge_values.data(), _buffer_merge.addr(),
                       sizeof(float) * num, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  try {
    // old mass

    auto m_t = thrust::device_pointer_cast(particles->get_mass_ptr());

    thrust::fill(thrust::device, _buffer_merged_last_step.addr(),
                 _buffer_merged_last_step.addr() + particles->size(), 0);

    DArray<float> old_masses(particles->size());
    CUDA_CALL(cudaMemcpy(old_masses.addr(), particles->get_mass_ptr(),
                         sizeof(float) * particles->size(),
                         cudaMemcpyDeviceToDevice));

    const float old_mass =
        thrust::reduce(m_t, m_t + num, 0, thrust::plus<float>());

    // Launch kernel with error checking
    cudaError_t err = cudaSuccess;

    // Run merge kernel
    if (t_merge_iter == 6) {
      merge_mark_gpu<<<(num + block_size - 1) / block_size, block_size>>>(
          num, particles->get_pos_ptr(), particles->get_mass_ptr(),
          particles->get_vel_ptr(), particles->get_surface_ptr(),
          particles->get_num_surface_ptr(), _buffer_remove.addr(),
          _buffer_merge.addr(), _buffer_merge_velocity.addr(),
          cell_start_granular.addr(), max_mass, cell_size, cell_length);

      t_merge_iter = 0;
    }
    t_merge_iter++;

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
                << std::endl;
      throw std::runtime_error("Kernel launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Kernel execution failed: " << cudaGetErrorString(err);
      // << std::endl;
      throw std::runtime_error("Kernel execution failed");
    }

    // Print info about particles being removed
    // std::vector<int> host_remove(num);
    // std::vector<float> host_mass(num);
    // CUDA_CALL(cudaMemcpy(host_remove.data(), _buffer_remove.addr(),
    //                      sizeof(int) * num, cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(host_mass.data(), particles->get_mass_ptr(),
    //                      sizeof(float) * num, cudaMemcpyDeviceToHost));

    // int removal_count = 0;
    // std::cout << "Particles marked for removal (mass):\n";
    // for (int i = 0; i < num; i++) {
    //   if (host_remove[i] == 1) {
    //     removal_count++;
    //     std::cout << "Particle " << i << ": mass = " << host_mass[i]
    //               << ", remove flag = " << host_remove[i]
    //               << ", merge mass delta = ";

    //     // Get merge mass delta for this particle
    //     float merge_delta;
    //     CUDA_CALL(cudaMemcpy(&merge_delta, &_buffer_merge.addr()[i],
    //                          sizeof(float), cudaMemcpyDeviceToHost));
    //     std::cout << merge_delta << "\n";
    //   }
    // }
    // std::cout << "Total particles marked for removal: " << removal_count
    //           << "\n";

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

    merge_count_gpu<<<(num + block_size - 1) / block_size, block_size>>>(
        num, _buffer_merge.addr(), particles->get_mass_ptr(),
        particles->get_vel_ptr(), _buffer_merge_count.addr(),
        _buffer_remove.addr(), _buffer_merge_velocity.addr(), _blend_factor);

    CUDA_CALL(cudaDeviceSynchronize());

    // // Print final state before removal
    // std::cout << "\nFinal state before removal:\n";
    // CUDA_CALL(cudaMemcpy(host_mass.data(), particles->get_mass_ptr(),
    //                      sizeof(float) * num, cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(host_remove.data(), _buffer_remove.addr(),
    //                      sizeof(int) * num, cudaMemcpyDeviceToHost));
    //
    // for (int i = 0; i < num; i++) {
    //   if (host_remove[i] == 1) {
    //     std::cout << "Particle " << i << ": final mass = " << host_mass[i]
    //               << "\n";
    //   }
    // }
    //
    // remove elements
    // Copy arrays from device to host for checking mass changes
    // std::vector<int> host_remove(particles->size());
    // std::vector<float> host_old_masses(particles->size());
    // std::vector<int> host_merge(particles->size());
    // std::vector<float> host_current_masses(particles->size());

    // CUDA_CALL(cudaMemcpy(host_remove.data(), _buffer_remove.addr(),
    //                      sizeof(int) * particles->size(),
    //                      cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(host_old_masses.data(), old_masses.addr(),
    //                      sizeof(float) * particles->size(),
    //                      cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(host_merge.data(), _buffer_merge.addr(),
    //                      sizeof(float) * particles->size(),
    //                      cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(host_current_masses.data(),
    // particles->get_mass_ptr(),
    //                      sizeof(float) * particles->size(),
    //                      cudaMemcpyDeviceToHost));

    // for (int i = 0; i < particles->size(); i++) {
    //   if (host_remove[i] == 0) {
    //     // Print masses that changed when they shouldn't have
    //     if (host_old_masses[i] != host_current_masses[i]) {
    //       std::cout << "Particle " << i << " mass changed unexpectedly from "
    //                 << host_old_masses[i] << " to " << host_current_masses[i]
    //                 << std::endl;
    //     }
    //   }
    // }

    CUDA_CALL(cudaDeviceSynchronize());

    particles->remove_elements(_buffer_remove);
    _buffer_merge_count.compact(_buffer_remove);
    _buffer_merge.compact(_buffer_remove);
    _buffer_remove.compact(_buffer_remove);

    // Get new size after compacting
    const int new_num = particles->size();

    // Get final state
    std::vector<float> final_masses(new_num);
    std::vector<float3> final_positions(new_num);
    std::vector<float3> final_velocities(new_num);

    CUDA_CALL(cudaMemcpy(final_masses.data(), particles->get_mass_ptr(),
                         sizeof(float) * new_num, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(final_positions.data(), particles->get_pos_ptr(),
                         sizeof(float3) * new_num, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(final_velocities.data(), particles->get_vel_ptr(),
                         sizeof(float3) * new_num, cudaMemcpyDeviceToHost));

    // // First, create a mapping of original to new indices
    // std::vector<int> index_mapping(num, -1); // Initialize all to -1
    // int new_idx = 0;
    // for (int i = 0; i < num; i++) {
    //   if (remove_flags[i] != 1) { // If particle survives
    //     index_mapping[i] = new_idx++;
    //   }
    // }

    // // Now verify ordering is maintained
    // bool ordering_maintained = true;
    // for (int old_idx = 0; old_idx < num; old_idx++) {
    //   if (remove_flags[old_idx] != 1) { // If this particle survived
    //     int new_idx = index_mapping[old_idx];

    //     // Check mass, accounting for merge values
    //     if (std::abs(final_masses[new_idx] - initial_masses[old_idx]) > 1e-6
    //     &&
    //         std::abs(final_masses[new_idx] - (initial_masses[old_idx] +
    //                                           merge_values[old_idx])) > 1e-6)
    //                                           {
    //       std::cout << "Mass mismatch for particle that moved from " <<
    //       old_idx
    //                 << " to " << new_idx << std::endl;
    //       std::cout << "Original mass: " << initial_masses[old_idx]
    //                 << std::endl;
    //       std::cout << "Merge value: " << merge_values[old_idx] << std::endl;
    //       std::cout << "Final mass: " << final_masses[new_idx] << std::endl;
    //       // ordering_maintained = false;
    //     }

    //     // Check position
    //     if (length(final_positions[new_idx] - initial_positions[old_idx]) >
    //         1e-6) {
    //       std::cout << "Position mismatch for particle that moved from "
    //                 << old_idx << " to " << new_idx << std::endl;
    //       std::cout << "Original position: (" << initial_positions[old_idx].x
    //                 << ", " << initial_positions[old_idx].y << ", "
    //                 << initial_positions[old_idx].z << ")" << std::endl;
    //       std::cout << "Final position: (" << final_positions[new_idx].x <<
    //       ", "
    //                 << final_positions[new_idx].y << ", "
    //                 << final_positions[new_idx].z << ")" << std::endl;
    //       ordering_maintained = false;
    //     }

    //     // Check velocity
    //     if (length(final_velocities[new_idx] - initial_velocities[old_idx]) >
    //         1e-6) {
    //       std::cout << "Velocity mismatch for particle that moved from "
    //                 << old_idx << " to " << new_idx << std::endl;
    //       std::cout << "Original velocity: (" <<
    //       initial_velocities[old_idx].x
    //                 << ", " << initial_velocities[old_idx].y << ", "
    //                 << initial_velocities[old_idx].z << ")" << std::endl;
    //       std::cout << "Final velocity: (" << final_velocities[new_idx].x
    //                 << ", " << final_velocities[new_idx].y << ", "
    //                 << final_velocities[new_idx].z << ")" << std::endl;
    //       ordering_maintained = false;
    //     }
    //   }
    // }

    // if (!ordering_maintained) {
    //   std::cout << "WARNING: Particle property consistency not maintained "
    //                "after compacting!"
    //             << std::endl;
    // } else {
    //   std::cout << "Particle property consistency maintained successfully."
    //             << std::endl;
    // }
    // add elements
    particles->add_elements(split_mass, split_pos, split_vel, n_split);
    _buffer_merge_count.resize(particles->size());
    _buffer_merge.resize(particles->size());
    _buffer_remove.resize(particles->size());

    cudaDeviceSynchronize();

    // change in mass

    // Print total mass after
    // auto m_t_n = thrust::device_pointer_cast(particles->get_mass_ptr());
    // const float new_mass = thrust::reduce(m_t_n, m_t_n + particles->size(),
    // 0,
    //                                       thrust::plus<float>());
    // std::cout << "Total mass after: " << new_mass << "\n";
    // if ((new_mass - old_mass) != 0) {
    //   std::cout << "Change in mass " << new_mass - old_mass << "\n";
    // }
    //

  } catch (const std::exception &e) {
    std::cerr << "Error in adaptive_sampling: " << e.what() << std::endl;
    return;
  }
}

__global__ void update_upsampled_cuda(
    float3 *pos_upsampled, float3 *pos_granular, float3 *pos_boundary,
    float3 *vel_granular, float3 *vel_upsampled, const int n,
    int *cell_start_upsampled, int *cell_start_granular,
    int *cell_start_boundary, const int3 cell_size, const float cell_length) {
  const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= n)
    return;

  const float3 pos_i = pos_upsampled[i];
  float3 weighted_vel = make_float3(0.0f, 0.0f, 0.0f);
  float total_weight = 0.0f;
  float max_w_ij = 0.0f;
  float alpha_i = 0.0f;
  const float3 g_t = make_float3(0.0f, -9.8f, 0.0f);
  const float d_t = 0.002f;

#pragma unroll
  for (auto m = 0; m < 27; __syncthreads(), ++m) {
    const auto cellID = particlePos2cellIdx(
        make_int3(pos_i / cell_length) +
            make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
        cell_size);

    if (cellID == (cell_size.x * cell_size.y * cell_size.z))
      continue;

    int j = cell_start_granular[cellID];
    while (j < cell_start_granular[cellID + 1]) {
      const float dis = length(pos_i - pos_granular[j]);
      const float t_1 = 1 - (dis * dis * r_9);

      const float w_ij = max(0.0f, t_1 * t_1 * t_1);
      total_weight += w_ij;

      weighted_vel += w_ij * vel_granular[j];

      if (w_ij > max_w_ij) {
        max_w_ij = w_ij;
      }
      j++;
    }

    int k = cell_start_boundary[cellID];
    while (k < cell_start_boundary[cellID + 1]) {
      const float dis = length(pos_i - pos_boundary[k]);
      const float t_1 = 1 - (dis * dis * r_9);

      const float w_ij = max(0.0f, t_1 * t_1 * t_1);
      total_weight += w_ij;

      if (w_ij > max_w_ij) {
        max_w_ij = w_ij;
      }
      // weighted_vel += w_ij * vel_granular[j];
      k++;
    }
  }

  if (total_weight == 0.0f) {
    vel_upsampled[i] = vel_upsampled[i] + g_t * d_t;
    pos_upsampled[i] = pos_upsampled[i] + vel_upsampled[i] * d_t;

    pos_upsampled[i] = pos_upsampled[i] + vel_upsampled[i] * d_t;
    if (pos_upsampled[i].y < 0.005f) {
      pos_upsampled[i].y = 0.005f;
    }
    if (pos_upsampled[i].x > 1.95) {
      pos_upsampled[i].x = 1.95;
    }
    if (pos_upsampled[i].x < 0.05) {
      pos_upsampled[i].x = 0.05;
    }
    if (pos_upsampled[i].z > 1.75) {
      pos_upsampled[i].z = 1.75;
    }
    if (pos_upsampled[i].z < 0.05) {
      pos_upsampled[i].z = 0.05;
    }
    return;
  }

  weighted_vel /= total_weight;
  if (max_w_ij <= 0.6023319616f || max_w_ij / total_weight >= 0.6) {
    alpha_i = 1 - max_w_ij;
  } else {
    alpha_i = 0.0f;
  }

  vel_upsampled[i] =
      alpha_i * (vel_upsampled[i] + g_t * d_t) + (1 - alpha_i) * weighted_vel;

  pos_upsampled[i] = pos_upsampled[i] + vel_upsampled[i] * d_t;
  if (pos_upsampled[i].y < 0.005f) {
    pos_upsampled[i].y = 0.005f;
  }
  if (pos_upsampled[i].x > 1.95) {
    pos_upsampled[i].x = 1.95;
  }
  if (pos_upsampled[i].x < 0.05) {
    pos_upsampled[i].x = 0.05;
  }
  if (pos_upsampled[i].z > 1.75) {
    pos_upsampled[i].z = 1.75;
  }
  if (pos_upsampled[i].z < 0.05) {
    pos_upsampled[i].z = 0.05;
  }

  return;
}

void Solver::upsampled_update(
    std::shared_ptr<GranularParticles> &particles,
    const std::shared_ptr<GranularParticles> &boundaries,
    std::shared_ptr<GranularParticles> &upsampled,
    const DArray<int> &cell_start_upsampled,
    const DArray<int> &cell_start_granular,
    const DArray<int> &cell_start_boundary, int3 cell_size, float3 space_size,
    float cell_length, const float density) {
  const int num = upsampled->size();

  update_upsampled_cuda<<<(num + block_size - 1) / block_size, block_size>>>(
      upsampled->get_pos_ptr(), particles->get_pos_ptr(),
      boundaries->get_pos_ptr(), particles->get_vel_ptr(),
      upsampled->get_vel_ptr(), num, cell_start_upsampled.addr(),
      cell_start_granular.addr(), cell_start_boundary.addr(), cell_size,
      cell_length);
}
