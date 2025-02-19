#pragma once
#include "GranularParticles.hpp"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

class Solver {
public:
  Solver(const std::shared_ptr<GranularParticles> &particles)
      : _max_iter(5), _blend_factor(40), _buffer_int(particles->size()),
        _buffer_float(particles->size()), _buffer_float3(particles->size()),
        _pos_t(particles->size()), _num_constraints(particles->size()),
        _buffer_remove(particles->size()), _buffer_split(particles->size()),
        _buffer_merge(particles->size()),
        _buffer_merge_velocity(particles->size()),
        _buffer_merge_count(particles->size()),
        _buffer_merge_lock(particles->size()) {

    thrust::device_ptr<int> thrust_remove =
        thrust::device_pointer_cast(_buffer_remove.addr());
    thrust::fill(thrust::device, thrust_remove,
                 thrust_remove + particles->size(), 0);

    thrust::device_ptr<float> thrust_merge =
        thrust::device_pointer_cast(_buffer_merge.addr());
    thrust::fill(thrust::device, thrust_merge, thrust_merge + particles->size(),
                 0);

    thrust::device_ptr<int> thrust_merge_count =
        thrust::device_pointer_cast(_buffer_merge_count.addr());
    thrust::fill(thrust::device, thrust_merge_count,
                 thrust_merge_count + particles->size(), 0);

    const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
    thrust::device_ptr<float3> thrust_merge_velocity =
        thrust::device_pointer_cast(_buffer_merge_velocity.addr());
    thrust::fill(thrust::device, thrust_merge_velocity,
                 thrust_merge_velocity + particles->size(), zero);
  }

  void step(std::shared_ptr<GranularParticles> &paticles,
            const std::shared_ptr<GranularParticles> &boundary,
            const DArray<int> &cell_start_particle,
            const DArray<int> &cell_start_boundary, float3 space_size,
            int3 cell_size, float cell_length, float dt, float3 G,
            const float density);
  ~Solver(){};
  void project(std::shared_ptr<GranularParticles> &particles,
               const std::shared_ptr<GranularParticles> &boundaries,
               const DArray<int> &cell_start_granular,
               const DArray<int> &cell_start_boundary, int3 cell_size,
               float3 space_size, float cell_length, int max_iter,
               const float density);
  void update_neighborhood(const std::shared_ptr<GranularParticles> &particles);
  void add_external_force(std::shared_ptr<GranularParticles> &particles,
                          float dt, float3 G);

  void update_particle_positions(std::shared_ptr<GranularParticles> &particles,
                                 float dt);

  void apply_mass_scaling(std::shared_ptr<GranularParticles> &particles);

  void final_update(std::shared_ptr<GranularParticles> &particles, float dt);

  void adaptive_sampling(std::shared_ptr<GranularParticles> &particles,
                         const std::shared_ptr<GranularParticles> &boundaries,
                         const DArray<int> &cell_start_granular,
                         const DArray<int> &cell_start_boundary,
                         const float max_mass, int3 cell_size,
                         float3 space_size, float cell_length,
                         const float density);
  void split(std::shared_ptr<GranularParticles> &particles);
  void distribute_mass(std::shared_ptr<GranularParticles> &particles);

  void upsampled_update(std::shared_ptr<GranularParticles> &particles,
                        const std::shared_ptr<GranularParticles> &boundaries,
                        std::shared_ptr<GranularParticles> &upsampled,
                        const DArray<int> &cell_start_upsampled,
                        const DArray<int> &cell_start_granular,
                        const DArray<int> &cell_start_boundary, int3 cell_size,
                        float3 space_size, float cell_length,
                        const float density);

  int *get_buffer_merge_count_ptr() const { return _buffer_merge_count.addr(); }
  float *get_buffer_merge_ptr() const { return _buffer_merge.addr(); }
  int *get_buffer_remove_ptr() const { return _buffer_remove.addr(); }

  void resize(const int size) {
    _buffer_merge_count.resize(size);
    _buffer_merge.resize(size);
    _buffer_remove.resize(size);
  }

  void trigger_explosion(std::shared_ptr<GranularParticles> &particles,
                         float explosion_force);

private:
  const int _max_iter;
  const int _blend_factor;
  DArray<int> _buffer_int;
  DArray<float> _buffer_float;
  DArray<float3> _buffer_float3;
  DArray<float3> _pos_t;
  DArray<int> _num_constraints;
  DArray<int> _buffer_remove;
  DArray<int> _buffer_split;
  DArray<float> _buffer_merge;
  DArray<float3> _buffer_merge_velocity;
  DArray<int> _buffer_merge_count;
  DArray<int> _buffer_merge_lock; // Add this member
};
