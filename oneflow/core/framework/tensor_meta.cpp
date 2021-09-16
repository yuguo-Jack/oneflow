#include "oneflow/core/framework/tensor_meta.h"

namespace oneflow {
namespace one {

MirroredTensorMeta::MirroredTensorMeta()
    : TensorMeta(std::make_shared<const Shape>(), DataType::kInvalidDataType),
      device_(Symbol<Device>()),
      stride_(std::make_shared<const Stride>()),
      storage_offset_(0) {}

MirroredTensorMeta::MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       Symbol<Device> device)
    : TensorMeta(shape, dtype),
      device_(device),
      stride_(std::make_shared<const Stride>(*shape)),
      storage_offset_(0) {}

bool MirroredTensorMeta::operator==(const MirroredTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && *this->device() == *other.device() && this->stride() == other.stride();
}

size_t MirroredTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Device>()(*device()) ^ std::hash<Stride>()(stride());
}

bool ConsistentTensorMeta::operator==(const ConsistentTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->nd_sbp() == other.nd_sbp() && this->parallel_desc() == other.parallel_desc();
}

size_t ConsistentTensorMeta::CalcHashValue() const {
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Symbol<cfg::NdSbp>>()(nd_sbp())
         ^ std::hash<Symbol<ParallelDesc>>()(parallel_desc());
}

}
}
