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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/embedding/persistent_table_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/cached_key_value_store.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

EmbeddingManager::~EmbeddingManager() {
  for (auto& pair : key_value_store_map_) { pair.second->SaveSnapshot("index"); }
}

embedding::KeyValueStore* EmbeddingManager::GetKeyValueStore(const std::string& embedding_name,
                                                             int64_t parallel_id) {
  OF_CUDA_CHECK(cudaSetDevice(parallel_id));
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  return it->second.get();
}

void EmbeddingManager::CreateKeyValueStore(
    const embedding::KeyValueStoreOptions& key_value_store_options, int64_t parallel_id,
    int64_t parallel_num) {
  OF_CUDA_CHECK(cudaSetDevice(parallel_id));
  const std::string& name = key_value_store_options.Name();
  const uint32_t line_size = key_value_store_options.LineSize();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& path = key_value_store_options.PersistentTablePath();
  const std::string& num_rank = std::to_string(parallel_num);
  const int32_t rank_id_suffix_length = num_rank.size();
  const std::string& rank_id = std::to_string(parallel_id);
  embedding::PersistentTableKeyValueStoreOptions options{};
  options.table_options.path = path + "/" + std::string(rank_id_suffix_length - rank_id.size(), '0')
                               + rank_id + "-" + num_rank;
  options.table_options.value_size = line_size * GetSizeOfDataType(DataType::kFloat);
  options.table_options.key_size = GetSizeOfDataType(DataType::kInt64);
  options.table_options.physical_block_size =
      key_value_store_options.PersistentTablePhysicalBlockSize();
  options.table_options.target_chunk_size_mb = 4 * 1024;
  store = NewPersistentTableKeyValueStore(options);
  const std::vector<embedding::CacheOptions> cache_options =
      key_value_store_options.GetCachesOptions();
  for (int i = 0; i < cache_options.size(); ++i) {
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options.at(i));
    LOG(ERROR) << "add cache: " << cache_options.at(i).policy << " "
               << cache_options.at(i).capacity;
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  key_value_store_map_.emplace(map_key, std::move(store));
}

void EmbeddingManager::SaveSnapshot(const std::string& embedding_name, int64_t parallel_id,
                                    const std::string& snapshot_name) {
  OF_CUDA_CHECK(cudaSetDevice(parallel_id));
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    it->second->SaveSnapshot(snapshot_name);
  } else {
    LOG(ERROR) << "Can not find embedding: " << embedding_name << "-" << parallel_id;
  }
}

void EmbeddingManager::LoadSnapshot(const std::string& embedding_name, int64_t parallel_id,
                                    const std::string& snapshot_name) {
  OF_CUDA_CHECK(cudaSetDevice(parallel_id));
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    if (it->second->SnapshotExists(snapshot_name)) {
      it->second->LoadSnapshot(snapshot_name);
    } else {
      LOG(ERROR) << "Here Exists Embedding name is: " << embedding_name << "-" << parallel_id
                 << " but no corresponding snapshot. ";
    }
  } else {
    LOG(ERROR) << "Can not find the embedding: " << embedding_name << "-" << parallel_id;
  }
}

}  // namespace oneflow
