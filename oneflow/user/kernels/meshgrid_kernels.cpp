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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MeshgridKernel final : public user_op::OpKernel {
 public:
  MeshgridKernel() = default;
  ~MeshgridKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
     const int64_t in_size = ctx->input_size("in");
     CHECK_EQ(ctx->output_size("out"), in_size);

     for (int64_t i = 0; i < in_size; ++i) {
         const user_op::Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
         user_op::Tensor* out_i = ctx->Tensor4ArgNameAndIndex("out", i);
         const ShapeView& shape = out_i->shape();

         for(int64_t j=0; j<shape.elem_cnt(); ++j){
            int64_t index_i = (j/shape.Count(i+1, shape.NumAxes()))%shape.At(i);
            out_i->mut_dptr<T>()[j] = in_i->dptr<T>()[index_i];
         }
     }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MESHGRID_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("meshgrid")                                                       \
      .SetCreateFn<MeshgridKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_MESHGRID_KERNEL(DeviceType::kCPU, float);
REGISTER_MESHGRID_KERNEL(DeviceType::kCPU, double);


template<DeviceType device_type, typename T>
class MeshgridGradKernel final : public user_op::OpKernel {
 public:
  MeshgridGradKernel() = default;
  ~MeshgridGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const int32_t index = ctx->Attr<int64_t>("index");
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    
    const ShapeView& shape = out->shape();
    for(int64_t i=0; i<shape.elem_cnt(); ++i){
          int64_t index_i = (i/shape.Count(i+1, shape.NumAxes()))%shape.At(i);
          dx->mut_dptr<T>()[index_i] += dy->dptr<T>()[i];
        }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MESHGRID_GRAD_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("meshgrid_grad")                                                       \
      .SetCreateFn<MeshgridGradKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_MESHGRID_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_MESHGRID_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
