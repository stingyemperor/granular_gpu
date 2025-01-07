#pragma once
#include "Global.hpp"
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class ParticleData {
public:
  ParticleData(thrust::host_vector<float3> &p) : _pos(p.begin(), p.end()) {
    _vel.reserve(p.size());
    _mass.reserve(p.size());
    _particle_to_cell.reserve(p.size());
  }

  ~ParticleData() {
    _pos.clear();
    _vel.clear();
    _mass.clear();
    _particle_to_cell.clear();

    _pos.shrink_to_fit();
    _vel.shrink_to_fit();
    _mass.shrink_to_fit();
    _particle_to_cell.shrink_to_fit();

    cudaDeviceSynchronize();
  }

  ParticleData(const ParticleData &) = delete;
  ParticleData &operator=(const ParticleData &) = delete;

  thrust::device_ptr<float3> get_pos_data() { return _pos.data(); }
  thrust::device_ptr<float3> get_vel_data() { return _vel.data(); }
  thrust::device_ptr<float> get_mass_data() { return _mass.data(); }

  float3 *get_pos_ptr() { return thrust::raw_pointer_cast(_pos.data()); }
  float3 *get_vel_ptr() { return thrust::raw_pointer_cast(_vel.data()); }
  float *get_mass_ptr() { return thrust::raw_pointer_cast(_mass.data()); }

  thrust::device_ptr<int> get_particle_to_cell_data() {
    return _particle_to_cell.data();
  }

  thrust::device_vector<float3> &get_pos() { return _pos; }
  thrust::device_vector<float> &get_mass() { return _mass; }

  int size() { return _pos.size(); }

  thrust::device_vector<float3> _pos;
  thrust::device_vector<float3> _vel;
  thrust::device_vector<float> _mass;
  thrust::device_vector<int> _particle_to_cell;
};
