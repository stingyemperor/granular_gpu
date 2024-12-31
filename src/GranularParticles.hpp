#pragma once
#include "Particles.hpp"

class GranularParticles final : public Particles {
public:
  explicit GranularParticles(const std::vector<float3> &p)
      : Particles(p), _mass(p.size()), _surface(p.size()),
        _particle_2_cell(p.size()) {
    CUDA_CALL(cudaMemcpy(_pos.addr(), &p[0], sizeof(float3) * p.size(),
                         cudaMemcpyHostToDevice));
  }

  GranularParticles(const GranularParticles &) = delete;
  GranularParticles &operator=(const GranularParticles &) = delete;

  int *get_particle_2_cell() const { return _particle_2_cell.addr(); }

  float *get_mass_ptr() const { return _mass.addr(); }
  int *get_surface_ptr() const { return _surface.addr(); }

  virtual ~GranularParticles() noexcept {}

protected:
  DArray<float> _mass;
  DArray<int> _surface;
  DArray<int> _particle_2_cell; // lookup key
};
