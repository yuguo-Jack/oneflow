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
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/variance_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class VarKernel final : public user_op::OpKernel {
 public:
  VarKernel() = default;
  ~VarKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const bool unbiased = ctx->Attr<bool>("unbiased");
    const T* in_ptr = input->dptr<T>();
    T* out_ptr = output->mut_dptr<T>();
    const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
    const int64_t input_dim_element = input->shape_view().elem_cnt();
    int64_t axis_dim_element = 1;
    for (int64_t i = 0; i < axis.size(); ++i) {
      axis_dim_element *= input->shape_view().At(axis[i]);
    }
    // when computing the variance with all the elements, the implementation of cuda kernel may use
    // tmp buffer for computation.
    T* tmp_buffer_ptr =
        (input_dim_element > 0
         && (axis.size() == input->shape_view().NumAxes() || input_dim_element == axis_dim_element)
         && DeviceType::kCUDA == device_type)
            ? ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr<T>()
            : nullptr;
    VarParamHelper param_helper(input->shape_view(), axis, unbiased);
    VarFunctor<device_type, T>()(ctx->stream(), in_ptr, out_ptr, tmp_buffer_ptr,
                                 param_helper.param);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_VAR_CPU_KERNEL(dtype)                                                           \
  REGISTER_USER_KERNEL("var").SetCreateFn<VarKernel<DeviceType::kCPU, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                             \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));
REGISTER_VAR_CPU_KERNEL(float)
REGISTER_VAR_CPU_KERNEL(double)
#undef REGISTER_VAR_CPU_KERNEL

#if defined(WITH_CUDA) || defined(WITH_ROCM)

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const Shape& input_shape = input.shape();
  const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
  const int64_t input_dim_element = input.shape().elem_cnt();
  int64_t axis_dim_element = 1;
  for (int64_t i = 0; i < axis.size(); ++i) { axis_dim_element *= input.shape().At(axis[i]); }
  if (input_dim_element > 0
      && (axis.size() == input_shape.NumAxes() || input_dim_element == axis_dim_element)) {
    return GetCudaAlignedSize(
        std::min(static_cast<int32_t>(std::ceil(std::sqrt(input.shape().elem_cnt()))),
                 kCudaMaxBlocksNum)
        * GetSizeOfDataType(input.data_type()) * 3);
  }
  return 0;
}

#define REGISTER_VAR_CUDA_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("var")                                                                 \
      .SetCreateFn<VarKernel<DeviceType::kCUDA, dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                        \
                       && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpBufferSize);

REGISTER_VAR_CUDA_KERNEL(float)
REGISTER_VAR_CUDA_KERNEL(double)
#undef REGISTER_VAR_CUDA_KERNEL
#endif

}  // namespace user_op
}  // namespace oneflow
