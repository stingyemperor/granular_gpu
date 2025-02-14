#pragma once
#include "DArray.hpp"
#include "Particles.hpp"

class GranularParticles final : public Particles {
public:
  explicit GranularParticles(const std::vector<float3> &p)
      : Particles(p), _mass(p.size()), _scaled_mass(p.size()),
        _surface(p.size()), _particle_2_cell(p.size()), _num_surface(p.size()),
        _is_animated(p.size()), _surface_distance(p.size()),
        _adaptive_last_step(p.size()) {
    CUDA_CALL(cudaMemcpy(_pos.addr(), &p[0], sizeof(float3) * p.size(),
                         cudaMemcpyHostToDevice));
  }

  GranularParticles(const GranularParticles &) = delete;
  GranularParticles &operator=(const GranularParticles &) = delete;

  int *get_particle_2_cell() const { return _particle_2_cell.addr(); }

  float *get_mass_ptr() const { return _mass.addr(); }
  float *get_scaled_mass_ptr() const { return _scaled_mass.addr(); }
  int *get_surface_ptr() const { return _surface.addr(); }
  float *get_surface_distance_ptr() const { return _surface_distance.addr(); }
  int *get_num_surface_ptr() const { return _num_surface.addr(); }
  int *get_adaptive_last_step_ptr() const { return _adaptive_last_step.addr(); }

  const DArray<float> &get_mass() const { return _mass; }

  void remove_elements(const DArray<int> &removal_flags) {
    try {
      _mass.compact(removal_flags);
    } catch (std::exception &e) {
      std::cerr << "Mismatch in mass: " << e.what() << std::endl;
    }

    try {
      // _surface.compact(removal_flags);
    } catch (std::exception &e) {
      std::cerr << "Mismatch in surface: " << e.what() << std::endl;
    }

    try {
      _particle_2_cell.compact(removal_flags);
    } catch (std::exception &e) {
      std::cerr << "Mismatch in particle_2_cell: " << e.what() << std::endl;
    }

    try {
      _adaptive_last_step.compact(removal_flags);
    } catch (std::exception &e) {
      std::cerr << "Mismatch in particle_2_cell: " << e.what() << std::endl;
    }

    Particles::remove_elements(removal_flags);
  }

  void add_elements(const DArray<float> &mass, const DArray<float3> &pos,
                    const DArray<float3> &vel, const int num) {
    // Validate input sizes
    if (mass.length() != num || pos.length() != num || vel.length() != num) {
      throw std::runtime_error(
          "Input array sizes don't match count in add_elements");
    }

    const unsigned int new_size = size() + num;

    // Resize all arrays if needed
    if (new_size > _mass.capacity()) {
      _mass.resize(new_size);
      // _surface.resize(new_size);
      _particle_2_cell.resize(new_size);
    }

    // Resize only the arrays we're actually using
    if (new_size > _mass.capacity()) {
      _mass.resize(new_size);
      // _surface.resize(new_size);
      _particle_2_cell.resize(new_size);
    }

    // Append to all arrays that get compacted
    _mass.append(mass);

    // Create temporary arrays for surface and particle_2_cell
    DArray<int> new_particle_2_cell(num);
    DArray<int> new_adaptive_last_step(num);

    thrust::fill(
        thrust::device, thrust::device_pointer_cast(new_particle_2_cell.addr()),
        thrust::device_pointer_cast(new_particle_2_cell.addr() + num), 0);

    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(new_adaptive_last_step.addr()),
        thrust::device_pointer_cast(new_adaptive_last_step.addr() + num), 1);

    // Append the temporary arrays
    // _surface.append(new_surface);
    _particle_2_cell.append(new_particle_2_cell);
    _adaptive_last_step.append(new_adaptive_last_step);

    // Call parent class add_elements
    Particles::add_elements(pos, vel);

    // Verify sizes
    const unsigned int final_size = size();
    if (_mass.length() != final_size ||
        _particle_2_cell.length() != final_size) {
      throw std::runtime_error("Array size mismatch after adding elements");
    }
  }

  int *get_is_animated_ptr() { return _is_animated.addr(); }

  virtual ~GranularParticles() noexcept {}

protected:
  DArray<float> _mass;
  DArray<float> _scaled_mass;
  DArray<int> _surface;
  DArray<float> _surface_distance;
  DArray<int> _num_surface;
  DArray<int> _particle_2_cell; // lookup key
  DArray<int> _is_animated;     // 1 if particle should be animated, 0 otherwise
  DArray<int> _adaptive_last_step;
};
