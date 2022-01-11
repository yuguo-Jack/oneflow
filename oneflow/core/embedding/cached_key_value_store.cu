/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/embedding/cached_key_value_store.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace embedding {

namespace {
template<typename Key, typename Elem>
__global__ void PostStoreGetKernel(uint32_t num_cache_missing, uint32_t num_store_missing,
                                   uint32_t num_elems_per_value,
                                   const uint32_t* cache_missing_indices,
                                   const uint32_t* store_missing_indices, const Elem* store_values,
                                   Elem* values, uint32_t* missing_indices) {
  const uint32_t num_cache_missing_elem = num_cache_missing * num_elems_per_value;
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_cache_missing_elem) {
    const uint32_t value_index = i / num_elems_per_value;
    const uint32_t elem_index = i - value_index * num_elems_per_value;
    values[cache_missing_indices[value_index] * num_elems_per_value + elem_index] = store_values[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_store_missing) {
    missing_indices[i] = cache_missing_indices[store_missing_indices[i]];
  }
}

template<typename Key, typename Elem>
class CacheKeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CacheKeyValueStoreImpl);
  CacheKeyValueStoreImpl(std::unique_ptr<KeyValueStore>&& store, std::unique_ptr<Cache>&& cache)
      : store_(std::move(store)), cache_(std::move(cache)), synced_(true) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    CHECK_EQ(store_->KeySize(), cache_->KeySize());
    CHECK_EQ(store_->ValueSize(), cache_->ValueSize());
    max_query_length_ = std::min(store_->MaxQueryLength(), cache_->MaxQueryLength());
    OF_CUDA_CHECK(cudaMalloc(&num_buffer_, sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_buffer_, sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMalloc(&keys_buffer_, max_query_length_ * store_->KeySize()));
    OF_CUDA_CHECK(cudaMalloc(&values_buffer_, max_query_length_ * store_->ValueSize()));
    OF_CUDA_CHECK(cudaMalloc(&indices_buffer0_, max_query_length_ * sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMalloc(&indices_buffer1_, max_query_length_ * sizeof(uint32_t)));
    num_elems_per_value_ = store_->ValueSize() / sizeof(Elem);
  }
  ~CacheKeyValueStoreImpl() {
    SyncToStore();
    OF_CUDA_CHECK(cudaFree(num_buffer_));
    OF_CUDA_CHECK(cudaFreeHost(host_num_buffer_));
    OF_CUDA_CHECK(cudaFree(keys_buffer_));
    OF_CUDA_CHECK(cudaFree(values_buffer_));
    OF_CUDA_CHECK(cudaFree(indices_buffer0_));
    OF_CUDA_CHECK(cudaFree(indices_buffer1_));
    cache_.reset();
    store_.reset();
  }

  uint32_t KeySize() const override { return store_->KeySize(); }
  uint32_t ValueSize() const override { return store_->ValueSize(); }
  uint32_t MaxQueryLength() const override { return max_query_length_; }
  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
           uint64_t* context) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values,
           uint64_t* context) override;
  void LoadSnapshot(const std::string& name) override;
  void SaveSnapshot(const std::string& name) override;

 private:
  void SyncToStore();

  std::unique_ptr<KeyValueStore> store_;
  std::unique_ptr<Cache> cache_;

  uint32_t* num_buffer_{};
  uint32_t* host_num_buffer_{};
  Key* keys_buffer_{};
  Elem* values_buffer_{};
  uint32_t* indices_buffer0_{};
  uint32_t* indices_buffer1_{};
  int device_index_{};
  uint32_t max_query_length_;
  uint32_t num_elems_per_value_{};
  std::mutex mutex_;
  bool synced_;
};

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::Get(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                            void* values, uint32_t* n_missing, void* missing_keys,
                                            uint32_t* missing_indices, uint64_t* context) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cuda_stream = stream->As<ep::CudaStream>();
  cache_->Get(stream, num_keys, keys, values, num_buffer_, keys_buffer_, indices_buffer0_);
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  const uint32_t num_cache_missing = *host_num_buffer_;
  if (num_cache_missing == 0) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    return;
  }
  store_->Get(stream, num_cache_missing, keys_buffer_, values_buffer_, n_missing, missing_keys,
              indices_buffer1_, context);
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, n_missing, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  const uint32_t num_store_missing = *host_num_buffer_;
  RUN_CUDA_KERNEL((PostStoreGetKernel<Key, Elem>), stream, num_cache_missing * num_elems_per_value_,
                  num_cache_missing, num_store_missing, num_elems_per_value_, indices_buffer0_,
                  indices_buffer1_, values_buffer_, static_cast<Elem*>(values), missing_indices);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::Put(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                            const void* values, uint64_t* context) {
  std::lock_guard<std::mutex> lock(mutex_);
  synced_ = false;
  auto cuda_stream = stream->As<ep::CudaStream>();
  cache_->Put(stream, num_keys, keys, values, num_buffer_, keys_buffer_, values_buffer_);
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  store_->Put(stream, *host_num_buffer_, keys_buffer_, values_buffer_, context);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::LoadSnapshot(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  store_->LoadSnapshot(name);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::SaveSnapshot(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  SyncToStore();
  store_->SaveSnapshot(name);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::SyncToStore() {
  if (synced_) { return; }
  CudaCurrentDeviceGuard guard(device_index_);
  auto device =
      Global<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, device_index_);
  CHECK(device);
  auto* stream = device->CreateStream();
  auto* cuda_stream = stream->As<ep::CudaStream>();
  uint64_t cache_capacity = cache_->Capacity();
  for (uint64_t start_key_index = 0; start_key_index < cache_capacity;
       start_key_index += max_query_length_) {
    cache_->Dump(stream, start_key_index,
                 std::min(start_key_index + max_query_length_, cache_capacity), num_buffer_,
                 keys_buffer_, values_buffer_);
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t),
                                  cudaMemcpyDefault, cuda_stream->cuda_stream()));
    CHECK_JUST(stream->Sync());
    if (*host_num_buffer_ == 0) { continue; }
    store_->Put(stream, *host_num_buffer_, keys_buffer_, values_buffer_, nullptr);
    CHECK_JUST(stream->Sync());
  }
  device->DestroyStream(stream);
  synced_ = true;
}

template<typename Key>
std::unique_ptr<KeyValueStore> DispatchElemType(std::unique_ptr<KeyValueStore>&& store,
                                                std::unique_ptr<Cache>&& cache) {
  const uint32_t value_size = store->ValueSize();
  if (value_size % sizeof(uint4) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint4>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint64_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint64_t>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint32_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint32_t>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint16_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint16_t>(std::move(store), std::move(cache)));
  } else {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint8_t>(std::move(store), std::move(cache)));
  }
}

std::unique_ptr<KeyValueStore> DispatchKeyType(std::unique_ptr<KeyValueStore>&& store,
                                               std::unique_ptr<Cache>&& cache) {
  const uint32_t key_size = store->KeySize();
  if (key_size == 4) {
    return DispatchElemType<uint32_t>(std::move(store), std::move(cache));
  } else if (key_size == 8) {
    return DispatchElemType<uint64_t>(std::move(store), std::move(cache));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<KeyValueStore> NewCachedKeyValueStore(std::unique_ptr<KeyValueStore>&& store,
                                                      std::unique_ptr<Cache>&& cache) {
  return DispatchKeyType(std::move(store), std::move(cache));
}

}  // namespace embedding

}  // namespace oneflow
