#pragma once
#include "GranularParticles.hpp"
#include "cuda_runtime.h"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

class Solver {
public:
  Solver(const std::shared_ptr<GranularParticles> &particles)
      : _max_iter(5), _blend_factor(4), _buffer_int(particles->size()),
        _buffer_float(particles->size()), _buffer_float3(particles->size()),
        _pos_t(particles->size()), _num_constraints(particles->size()),
        _buffer_remove(particles->size()), _buffer_split(particles->size()),
        _buffer_merge(particles->size()),
        _buffer_merge_count(particles->size()) {}

  void step(std::shared_ptr<GranularParticles> &paticles,
            const std::shared_ptr<GranularParticles> &boundary,
            const DArray<int> &cell_start_particle,
            const DArray<int> &cell_start_boundary, float3 space_size,
            int3 cell_size, float cell_length, float dt, float3 G,
            const float density);
  ~Solver() {};
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

  void final_update(std::shared_ptr<GranularParticles> &particles, float dt);

  void adaptive_sampling(std::shared_ptr<GranularParticles> &particles,
                         const DArray<int> &cell_start_granular,
                         const float max_mass, int3 cell_size,
                         float3 space_size, float cell_length,
                         const float density);
  void split(std::shared_ptr<GranularParticles> &particles);

  int *get_buffer_merge_count_ptr() const { return _buffer_merge_count.addr(); }
  float *get_buffer_merge_ptr() const { return _buffer_merge.addr(); }
  int *get_buffer_remove_ptr() const { return _buffer_remove.addr(); }

  void resize(const int size) {
    _buffer_merge_count.resize(size);
    _buffer_merge.resize(size);
    _buffer_remove.resize(size);
  }

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
  DArray<int> _buffer_merge_count;
};
