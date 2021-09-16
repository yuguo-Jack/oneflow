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
#ifndef ONEFLOW_FRAMEWORK_TENSOR_META_H_
#define ONEFLOW_FRAMEWORK_TENSOR_META_H_

#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {
namespace cfg {

class NdSbp;
}

class Shape;
class Device;
class Stride;
class ParallelDesc;

namespace one {

class MirroredTensorMeta;
class ConsistentTensorMeta;

class TensorMeta : public user_op::TensorDesc {
 public:
  TensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype)
      : shape_(shape), data_type_(dtype), is_dynamic_(false) {}
  TensorMeta(const TensorMeta&) = default;
  TensorMeta(TensorMeta&&) = default;
  virtual ~TensorMeta() = default;

  const std::shared_ptr<const Shape>& shape_ptr() const { return shape_; }

  const Shape& shape() const override { return *shape_; }
  DataType dtype() const { return data_type_; }
  DataType data_type() const override { return data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }

  void set_shape(const std::shared_ptr<const Shape>& val) { shape_ = val; }
  Shape* mut_shape() override { return const_cast<Shape*>(shape_.get()); }
  DataType* mut_dtype() { return &data_type_; }
  void set_dtype(DataType data_type) { data_type_ = data_type; }
  DataType* mut_data_type() override { return &data_type_; }
  bool* mut_is_dynamic() override { return &is_dynamic_; }
  void set_is_dynamic(bool val) override { is_dynamic_ = val; }

  virtual size_t CalcHashValue() const = 0;
  virtual bool Equals(const TensorMeta& other) const = 0; 
  virtual bool EqualsMirroredTensorMeta(const MirroredTensorMeta& other) const = 0; 
  virtual bool EqualsConsistentTensorMeta(const ConsistentTensorMeta& other) const = 0; 

 private:
  std::shared_ptr<const Shape> shape_;
  DataType data_type_;
  bool is_dynamic_;
};

class MirroredTensorMeta : public TensorMeta {
 public:
  // uninitialized MirroredTensorMeta.
  MirroredTensorMeta();
  MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                     Symbol<Device> device);
  virtual ~MirroredTensorMeta() = default;

  const Symbol<Device>& device() const { return device_; }
  const Stride& stride() const { return *stride_; }
  const std::shared_ptr<const Stride>& stride_ptr() const { return stride_; }
  int64_t storage_offset() const { return storage_offset_; }

  Symbol<Device>* mut_device() { return &device_; }
  void set_stride(const std::shared_ptr<const Stride>& stride) { stride_ = stride; }
  void set_storage_offset(int64_t offset) { storage_offset_ = offset; }

  bool operator==(const MirroredTensorMeta& other) const;
  size_t CalcHashValue() const override;
  bool Equals(const TensorMeta& other) const override {
    return other.EqualsMirroredTensorMeta(*this);
  }
  bool EqualsMirroredTensorMeta(const MirroredTensorMeta& other) const override {
    return *this == other;
  }
  bool EqualsConsistentTensorMeta(const ConsistentTensorMeta& other) const override {
    return false;
  }

 private:
  Symbol<Device> device_;
  std::shared_ptr<const Stride> stride_;
  int64_t storage_offset_;
};

class ConsistentTensorMeta : public TensorMeta {
 public:
  ConsistentTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                       Symbol<cfg::NdSbp> nd_sbp, Symbol<ParallelDesc> parallel_desc)
      : TensorMeta(shape, dtype), nd_sbp_(nd_sbp), parallel_desc_(parallel_desc) {}
  ConsistentTensorMeta(const ConsistentTensorMeta&) = default;
  ConsistentTensorMeta(ConsistentTensorMeta&&) = default;
  virtual ~ConsistentTensorMeta() = default;

  bool operator==(const ConsistentTensorMeta& other) const;

  Symbol<cfg::NdSbp> nd_sbp() const { return nd_sbp_; }
  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

  void set_nd_sbp(Symbol<cfg::NdSbp> val) { nd_sbp_ = val; }

  void set_parallel_desc(Symbol<ParallelDesc> val) { parallel_desc_ = val; }

  size_t CalcHashValue() const override;
  bool Equals(const TensorMeta& other) const override {
    return other.EqualsConsistentTensorMeta(*this);
  }
  bool EqualsMirroredTensorMeta(const MirroredTensorMeta& other) const override {
    return false;
  }
  bool EqualsConsistentTensorMeta(const ConsistentTensorMeta& other) const override {
    return *this == other;
  }

 private:
  Symbol<cfg::NdSbp> nd_sbp_;
  Symbol<ParallelDesc> parallel_desc_;
};

class MutMirroredTensorMeta : public TensorMeta {
 public:
  MutMirroredTensorMeta() : TensorMeta(std::make_shared<const Shape>(), kInvalidDataType) {}
  MutMirroredTensorMeta(const MutMirroredTensorMeta&) = default;
  MutMirroredTensorMeta(MutMirroredTensorMeta&&) = default;
  ~MutMirroredTensorMeta() override = default;

  size_t CalcHashValue() const override {
    UNIMPLEMENTED();
    return 0;
  }
  bool Equals(const TensorMeta& other) const override {
    UNIMPLEMENTED();
    return false;
  }
  bool EqualsMirroredTensorMeta(const MirroredTensorMeta& other) const override {
    UNIMPLEMENTED();
    return false;
  }
  bool EqualsConsistentTensorMeta(const ConsistentTensorMeta& other) const override {
    UNIMPLEMENTED();
    return false;
  }
};

class TensorMetaHashKey final {
 public:
  explicit TensorMetaHashKey(const std::shared_ptr<TensorMeta>& tensor_meta)
      : tensor_meta_(tensor_meta), hash_value_(tensor_meta->CalcHashValue()) {}
  TensorMetaHashKey(const TensorMetaHashKey&&) = default;
  TensorMetaHashKey(TensorMetaHashKey&&&) = default;
  ~TensorMetaHashKey() = default;

  bool operator==(const TensorMetaHashKey& other) const {
    return tensor_meta_->Equals(*other.tensor_meta_);
  }

  size_t hash_value() const { return hash_value_; }

 private:
  std::shared_ptr<TensorMeta> tensor_meta_;
  const size_t hash_value_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::MirroredTensorMeta> final {
  size_t operator()(const oneflow::one::MirroredTensorMeta& local_tensor_meta) const {
    return local_tensor_meta.CalcHashValue();
  }
};

template<>
struct hash<oneflow::one::ConsistentTensorMeta> final {
  size_t operator()(const oneflow::one::ConsistentTensorMeta& consistent_tensor_meta) const {
    return consistent_tensor_meta.CalcHashValue();
  }
};

template<>
struct hash<oneflow::one::TensorMetaHashKey> final {
  size_t operator()(const oneflow::one::TensorMetaHashKey& key) const {
    return key.hash_value();
  }
};

}  // namespace std

#endif  // ONEFLOW_FRAMEWORK_TENSOR_META_H_
