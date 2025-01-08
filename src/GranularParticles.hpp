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

  const DArray<float> &get_mass() const { return _mass; }

  void remove_elements(const DArray<int> &removal_flags) {
    _mass.compact(removal_flags);
    _surface.compact(removal_flags);
    _particle_2_cell.compact(removal_flags);

    Particles::remove_elements(removal_flags);
  }

  void add_elements(const DArray<float> &mass, const DArray<float3> &pos,
                    const DArray<float3> &vel, const int num) {
    _mass.append(mass);
    Particles::add_elements(pos, vel);
  }

  virtual ~GranularParticles() noexcept {}

protected:
  DArray<float> _mass;
  DArray<int> _surface;
  DArray<int> _particle_2_cell; // lookup key
};
