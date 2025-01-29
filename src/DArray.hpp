
#pragma once
#include "Global.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

template <typename T> class DArray {
public:
  explicit DArray(unsigned int size, unsigned int initial_capacity = 0)
      : _length(size), _capacity(std::max(size, initial_capacity)) {

    // Add size validation
    if (size > 1000000000) { // Sanity check for extremely large allocations
      throw std::runtime_error("Suspiciously large array size requested");
    }

    // Check available memory before allocation
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_mem = sizeof(T) * _capacity;

    if (required_mem > free_mem) {
      std::cerr << "Not enough GPU memory. Required: " << required_mem
                << " Available: " << free_mem << std::endl;
      throw std::runtime_error("Insufficient GPU memory");
    }

    // Try allocation with error checking
    cudaError_t err = cudaMalloc(&_addr, required_mem);
    if (err != cudaSuccess) {
      std::cerr << "CUDA malloc failed for size " << size << " capacity "
                << _capacity << " bytes " << required_mem
                << " Error: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("CUDA malloc failed");
    }

    // Initialize memory to zero
    err = cudaMemset(_addr, 0, required_mem);
    if (err != cudaSuccess) {
      cudaFree(_addr);
      throw std::runtime_error("CUDA memset failed");
    }
  }

  ~DArray() {
    if (_addr) {
      cudaError_t err = cudaFree(_addr);
      if (err != cudaSuccess) {
        std::cerr << "Failed to free CUDA memory in destructor: "
                  << cudaGetErrorString(err) << std::endl;
      }
      _addr = nullptr;
    }
  }

  // Basic accessors
  unsigned int length() const { return _length; }
  unsigned int capacity() const { return _capacity; }
  T *addr() const { return _addr; }

  // Compact array using removal flags - Thrust optimized version
  unsigned int compact(const DArray<int> &removal_flags) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (removal_flags.length() != _length) {
      std::cerr << "Removal flags length (" << removal_flags.length()
                << ") doesn't match array length (" << _length << ")"
                << std::endl;
      throw std::runtime_error("Removal flags length mismatch");
    }

    try {
      thrust::device_ptr<T> data_ptr(_addr);
      thrust::device_ptr<const int> flags_ptr(removal_flags.addr());

      // Create temporary storage
      thrust::device_vector<T> temp(_length);

      // Count elements to keep (where flag != 1)
      int keep_count =
          thrust::count_if(thrust::device, flags_ptr, flags_ptr + _length,
                           [] __device__(int flag) { return flag != 1; });

      // Copy elements where flag != 1 (keeping both 0 and -1)
      auto new_end = thrust::copy_if(
          thrust::device, data_ptr, data_ptr + _length, flags_ptr, temp.begin(),
          [] __device__(int flag) { return flag != 1; });

      unsigned int new_length =
          static_cast<unsigned int>(new_end - temp.begin());

      if (new_length != keep_count) {
        std::cerr << "Inconsistent compact results: expected " << keep_count
                  << " but got " << new_length << std::endl;
        throw std::runtime_error("Inconsistent compact results");
      }

      // Copy back compacted data
      if (new_length < _length) {
        cudaError_t err =
            cudaMemcpy(_addr, thrust::raw_pointer_cast(temp.data()),
                       sizeof(T) * new_length, cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
          std::cerr << "Error copying back compacted data: "
                    << cudaGetErrorString(err) << std::endl;
          throw std::runtime_error("Compact copy failed");
        }
      }

      _length = new_length;
      return _length;

    } catch (const thrust::system_error &e) {
      std::cerr << "Thrust error in compact: " << e.what() << std::endl;
      throw std::runtime_error(std::string("Thrust error: ") + e.what());
    } catch (const std::exception &e) {
      std::cerr << "Error in compact: " << e.what() << std::endl;
      throw;
    }
  }

  // Append elements from another DArray
  void append(const DArray<T> &other) {
    const unsigned int new_length = _length + other.length();

    // Resize if necessary
    if (new_length > _capacity) {
      resize(std::max(new_length, _capacity * 2));
    }

    // Create thrust device pointers
    thrust::device_ptr<T> dst_ptr(_addr + _length);
    thrust::device_ptr<const T> src_ptr(other.addr());

    // Copy the new elements
    thrust::copy(thrust::device, src_ptr, src_ptr + other.length(), dst_ptr);

    _length = new_length;
  }

  // Remove elements within a range
  void remove(unsigned int start, unsigned int count) {
    if (start + count > _length) {
      throw std::runtime_error("Remove range exceeds array bounds");
    }

    if (start + count < _length) {
      // Create thrust device pointers
      thrust::device_ptr<T> data_ptr(_addr);

      // Move remaining elements to fill the gap
      thrust::copy(thrust::device,
                   data_ptr + start + count, // Source start
                   data_ptr + _length,       // Source end
                   data_ptr + start          // Destination start
      );
    }

    _length -= count;
  }

  static void verifyArrayOrdering(const std::vector<DArray<T> *> &arrays,
                                  const DArray<int> &removal_flags,
                                  const char *arrayNames[]) {
    if (arrays.empty())
      return;

    const unsigned int length = arrays[0]->length();

    // Verify all arrays have same length
    for (size_t i = 1; i < arrays.size(); i++) {
      if (arrays[i]->length() != length) {
        throw std::runtime_error(std::string("Array length mismatch between ") +
                                 arrayNames[0] + " and " + arrayNames[i]);
      }
    }

    // Create host vectors for verification
    std::vector<std::vector<T>> initial_data(arrays.size());
    std::vector<std::vector<T>> final_data(arrays.size());
    std::vector<int> remove_flags(length);

    // Copy initial data and removal flags to host
    for (size_t i = 0; i < arrays.size(); i++) {
      initial_data[i].resize(length);
      CUDA_CALL(cudaMemcpy(initial_data[i].data(), arrays[i]->addr(),
                           sizeof(T) * length, cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaMemcpy(remove_flags.data(), removal_flags.addr(),
                         sizeof(int) * length, cudaMemcpyDeviceToHost));

    // Perform compaction
    for (auto arr : arrays) {
      arr->compact(removal_flags);
    }

    // Copy final data to host
    const unsigned int new_length = arrays[0]->length();
    for (size_t i = 0; i < arrays.size(); i++) {
      if (arrays[i]->length() != new_length) {
        throw std::runtime_error("Inconsistent lengths after compaction");
      }
      final_data[i].resize(new_length);
      CUDA_CALL(cudaMemcpy(final_data[i].data(), arrays[i]->addr(),
                           sizeof(T) * new_length, cudaMemcpyDeviceToHost));
    }

    // Verify ordering is maintained
    std::vector<int> surviving_indices;
    for (unsigned int i = 0; i < length; i++) {
      if (remove_flags[i] != 1) {
        surviving_indices.push_back(i);
      }
    }

    for (unsigned int i = 0; i < new_length; i++) {
      int old_idx = surviving_indices[i];
      for (size_t arr_idx = 0; arr_idx < arrays.size(); arr_idx++) {
        if (memcmp(&final_data[arr_idx][i], &initial_data[arr_idx][old_idx],
                   sizeof(T)) != 0) {
          std::cerr << "Order mismatch in " << arrayNames[arr_idx]
                    << " at index " << i << " (original index " << old_idx
                    << ")" << std::endl;
          throw std::runtime_error(
              "Array ordering not maintained during compaction");
        }
      }
    }
  }

public:
  T *_addr;
  unsigned int _length;
  unsigned int _capacity;

  void resize(unsigned int new_capacity) {
    T *new_addr;
    CUDA_CALL(cudaMalloc(&new_addr, sizeof(T) * new_capacity));

    // Create thrust device pointers
    thrust::device_ptr<T> dst_ptr(new_addr);
    thrust::device_ptr<T> src_ptr(_addr);

    // Copy existing data
    if (_length > 0) {
      thrust::copy(thrust::device, src_ptr,
                   src_ptr + std::min(_length, new_capacity), dst_ptr);
    }

    cudaFree(_addr);
    _addr = new_addr;
    _capacity = new_capacity;
    _length = std::min(_length, new_capacity);
  }
};
