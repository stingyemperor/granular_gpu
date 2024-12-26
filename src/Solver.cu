#include "CUDAFunctions.cuh"
#include "Global.hpp"
#include "Solver.hpp"
#include "cuda_runtime.h"
#include "helper_math.h"
#include <vector_types.h>

void Solver::step(std::shared_ptr<GranularParticles> &particles,
                  const std::shared_ptr<GranularParticles> &boundary,
                  const DArray<int> &cell_start_particle,
                  const DArray<int> &cell_start_boundary, float3 space_size,
                  int3 cell_size, float cell_length, float dt, float3 G) {

  update_neighborhood(particles);

  // apply forces
  // update velocity
  add_external_force(particles, dt, G);
  update_particle_positions(particles, dt);
}

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

struct update_position_functor {
  float dt;

  update_position_functor(float _dt) : dt(_dt) {}

  __host__ __device__ float3 operator()(const thrust::tuple<float3, float3> &t) const {
    const float3 &pos = thrust::get<0>(t);
    const float3 &vel = thrust::get<1>(t);
    return make_float3(pos.x + dt * vel.x, pos.y + dt * vel.y, pos.z + dt * vel.z);
  }
};


void Solver::update_particle_positions(
    std::shared_ptr<GranularParticles> &particles, float dt) {
  // Assuming particles->get_pos_ptr() returns a pointer to the first element of
  // the position buffer and particles->get_vel_ptr() returns a pointer to the
  // first element of the velocity buffer

  // Create zip iterator for positions and velocities
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr(), particles->get_vel_ptr()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(particles->get_pos_ptr() + particles->size(),
                         particles->get_vel_ptr() + particles->size()));

  // Update positions by applying the 'update_position_functor' across the range
  thrust::transform(thrust::device, begin, end,
                    particles->get_pos_ptr(), // Output to the positions buffer
                    update_position_functor(dt));
}
