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
      : _max_iter(20), _buffer_int(particles->size()),
        _buffer_float(particles->size()), _buffer_float3(particles->size()) {}
  void step(std::shared_ptr<GranularParticles> &paticles,
            const std::shared_ptr<GranularParticles> &boundary,
            const DArray<int> &cell_start_fluid,
            const DArray<int> &cell_start_boundary, float3 space_size,
            int3 cell_size, float cell_length, float radius, float dt,
            float3 G);
  ~Solver() {};
  void update_neighborhood(const std::shared_ptr<GranularParticles> &particles);
  void add_external_force(std::shared_ptr<GranularParticles> &particles,
                          float dt, float3 G);

private:
  const int _max_iter;
  DArray<int> _buffer_int;
  DArray<float> _buffer_float;
  DArray<float3> _buffer_float3;
};
