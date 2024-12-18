#pragma once
#include "DArray.hpp"
#include "Global.hpp"
#include <vector>

class Particles {
public:
Particles::Particles(const std::vector<float3>& p)
    :pos(p.size()), vel(p.size()) {
    CUDA_CALL(cudaMemcpy(pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
}

Particles(const Particles&) = delete;
Particles& operator=(const Particles&) = delete;

unsigned int size() const {
    return pos.length();
}
float3* getPosPtr() const {
    return pos.addr();
}
float3* getVelPtr() const {
    return vel.addr();
}
const DArray<float3>& getPos() const {
    return pos;
}

protected:
    DArray<float3> pos;
    DArray<float3> vel;
    DArray<float> mass;
    DArray<int> particle2Cell; // lookup key
};