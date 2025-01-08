#pragma once
#include "DArray.hpp"
#include "Global.hpp"
#include <thrust/device_vector.h>
#include <vector>

class Particles {
public:
  explicit Particles(const std::vector<float3> &p)
      : _pos(p.size()), _vel(p.size()) {
    CUDA_CALL(cudaMemcpy(_pos.addr(), &p[0], sizeof(float3) * p.size(),
                         cudaMemcpyHostToDevice));
  }

  Particles(const Particles &) = delete;
  Particles &operator=(const Particles &) = delete;

  unsigned int size() const { return _pos.length(); }
  float3 *get_pos_ptr() const { return _pos.addr(); }
  float3 *get_vel_ptr() const { return _vel.addr(); }
  const DArray<float3> &get_pos() const { return _pos; }

  void remove_elements(const DArray<int> &removal_flags) {
    _pos.compact(removal_flags);
    _vel.compact(removal_flags);
  }

  void add_elements(const DArray<float3> &pos, const DArray<float3> &vel) {
    _pos.append(pos);
    _vel.append(vel);
  }

  virtual ~Particles() noexcept {}

public:
  DArray<float3> _pos;
  DArray<float3> _vel;
  thrust::device_vector<float3> _position;
  thrust::device_vector<float3> _velocity;
};
