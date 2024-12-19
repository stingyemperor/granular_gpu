#pragma once
#include "DArray.hpp"
#include "Global.hpp"
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

  virtual ~Particles() noexcept {}

protected:
  DArray<float3> _pos;
  DArray<float3> _vel;
};
