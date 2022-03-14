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
#ifndef ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
#define ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
#include "nlohmann/json.hpp"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/embedding/cache.h"

namespace oneflow {
namespace embedding {

namespace {

void ParseCacheOptions(const nlohmann::json& cache_obj, embedding::CacheOptions* cache_options) {
  CHECK_GT(cache_options->key_size, 0);
  CHECK_GT(cache_options->value_size, 0);
  CHECK(cache_obj.contains("policy"));
  CHECK(cache_obj["policy"].is_string());
  std::string policy = cache_obj["policy"].get<std::string>();
  if (policy == "lru") {
    cache_options->policy = embedding::CacheOptions::Policy::kLRU;
  } else if (policy == "full") {
    cache_options->policy = embedding::CacheOptions::Policy::kFull;
  } else {
    UNIMPLEMENTED();
  }
  CHECK(cache_obj.contains("cache_memory_budget_mb"));
  CHECK(cache_obj["cache_memory_budget_mb"].is_number());
  cache_options->capacity =
      cache_obj["cache_memory_budget_mb"].get<int64_t>() * 1024 * 1024 / cache_options->value_size;
  CHECK(cache_obj.contains("value_memory_kind"));
  CHECK(cache_obj["value_memory_kind"].is_string());
  std::string value_memory_kind = cache_obj["value_memory_kind"].get<std::string>();
  if (value_memory_kind == "device") {
    cache_options->value_memory_kind = embedding::CacheOptions::MemoryKind::kDevice;
  } else if (value_memory_kind == "host") {
    cache_options->value_memory_kind = embedding::CacheOptions::MemoryKind::kHost;
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

class KeyValueStoreOptions final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreOptions);
  KeyValueStoreOptions(std::string json_serialized) {
    auto json_object = nlohmann::json::parse(json_serialized);
    auto GetValue = [](const nlohmann::json& obj, const std::string& attr) -> nlohmann::json {
      nlohmann::json val = obj[attr];
      if (val == nlohmann::detail::value_t::null) { UNIMPLEMENTED(); }
      return val;
    };
    name_ = GetValue(json_object, "name");
    const int64_t storage_dim = GetValue(json_object, "storage_dim");
    line_size_ = storage_dim;
    auto kv_store = GetValue(json_object, "kv_store");
    auto caches = kv_store["caches"];
    if (caches != nlohmann::detail::value_t::null && caches.size() > 0) {
      CHECK(caches.is_array());
      cache_options_.resize(caches.size());
      for (int i = 0; i < caches.size(); ++i) {
        cache_options_.at(i).key_size = GetSizeOfDataType(DataType::kInt64);
        cache_options_.at(i).value_size = GetSizeOfDataType(DataType::kFloat) * line_size_;
        ParseCacheOptions(caches.at(i), &cache_options_.at(i));
      }
    }
    if (kv_store["persistent_table"] != nlohmann::detail::value_t::null) {
      auto persistent_table = kv_store["persistent_table"];
      persistent_table_path_ = GetValue(persistent_table, "path");
      persistent_table_phisical_block_size_ = GetValue(persistent_table, "physical_block_size");
    } else {
      UNIMPLEMENTED();
    }
  }
  ~KeyValueStoreOptions() = default;

  std::string Name() const { return name_; }
  int64_t LineSize() const { return line_size_; }
  std::vector<CacheOptions> GetCachesOptions() const { return cache_options_; }
  std::string PersistentTablePath() const { return persistent_table_path_; }
  int64_t PersistentTablePhysicalBlockSize() const { return persistent_table_phisical_block_size_; }
  bool IsFullCache() const {
    if (cache_options_.size() > 0 && cache_options_.at(0).policy == CacheOptions::Policy::kFull) {
      return true;
    }
    return false;
  }

 private:
  std::string name_;
  int64_t line_size_;
  std::string l1_cache_policy_;
  int64_t l1_cache_memory_budget_mb_;
  std::string l1_cache_value_memory_kind_;
  std::string l2_cache_policy_;
  int64_t l2_cache_memory_budget_mb_;
  std::string l2_cache_value_memory_kind_;
  std::string persistent_table_path_;
  int64_t persistent_table_phisical_block_size_;
  std::vector<CacheOptions> cache_options_;
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
