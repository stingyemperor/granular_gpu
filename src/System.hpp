#include "ParticleData.hpp"
#include <memory>
#include <thrust/device_vector.h>

class System {
public:
  System(std::shared_ptr<ParticleData> granular_particles,
         std::shared_ptr<ParticleData> boundary_particles,
         const float3 space_size, const float cell_length, const float dt,
         const float3 g, const int3 cell_size, const int density);

  System(const System &) = delete;
  System &operator=(const System &) = delete;

  auto get_granular_particles() const {
    return static_cast<const std::shared_ptr<ParticleData>>(
        _granular_particles);
  }

  auto get_boundary_particles() const {
    return static_cast<const std::shared_ptr<ParticleData>>(
        _boundary_particles);
  }

  thrust::device_ptr<int> get_surface_data() { return _surface.data(); }
  int *get_surface_ptr() { return thrust::raw_pointer_cast(_surface.data()); }

  void step();
  void neighbor_search(const std::shared_ptr<ParticleData> &granular_particles,
                       thrust::device_vector<int> &cell_start);

  void
  set_surface_particles(const std::shared_ptr<ParticleData> &granular_particles,
                        thrust::device_vector<int> &cell_start);

  int size() const { return (*_granular_particles).size(); }
  int boundary_size() { return (*_boundary_particles).size(); }
  int total_size() const {
    return (*_granular_particles).size() + (*_boundary_particles).size();
  }

  const std::shared_ptr<ParticleData> &_granular_particles;
  const std::shared_ptr<ParticleData> &_boundary_particles;

  thrust::device_vector<int> _surface;
  thrust::device_vector<int> _cell_start_granular;
  thrust::device_vector<int> _cell_start_boundary;
  const float _cell_length;
  const float3 _space_size;
  const float _dt;
  const float3 _g;
  const int _density;
  const int _max_mass;
  const int _min_mass;
  const int3 _cell_size;
  thrust::device_vector<int> _boundary_t;
  thrust::device_vector<int> _cell_index_t;
};
