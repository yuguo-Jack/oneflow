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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace {

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<int N>
__global__ void ComputeNchwPackHalf(const int32_t elem_cnt, const int32_t inner_dim_size,
                                    const int32_t param_size, const float epsilon,
                                    const Pack<half, N>* dy, const float* gamma,
                                    const float* moving_variance, Pack<half, N>* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    Pack<half, N> dy_pack = dy[i];
    Pack<half, N> dx_pack;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const int32_t param_idx = (i * N + j) / inner_dim_size % param_size;
      const float inv_variance = rsqrt(moving_variance[param_idx] + epsilon);
      dx_pack.elem[j] =
          static_cast<float>(static_cast<float>(dy_pack.elem[j]) * gamma[param_idx] * inv_variance);
    }
    dx[i] = dx_pack;
  }
}

template<typename T, typename ComputeType>
__global__ void ComputeNchw(const int32_t elem_cnt, const int32_t inner_dim_size,
                            const int32_t param_size, const float epsilon, const T* dy,
                            const ComputeType* gamma, const ComputeType* moving_variance, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t param_idx = i / inner_dim_size % param_size;
    const ComputeType inv_variance = rsqrt(moving_variance[param_idx] + epsilon);
    dx[i] = static_cast<T>(static_cast<ComputeType>(dy[i]) * gamma[param_idx] * inv_variance);
  }
}

template<typename T, typename ComputeType>
__global__ void ComputeNhwc(const int32_t elem_cnt, const int32_t param_size, const float epsilon,
                            const T* dy, const ComputeType* gamma,
                            const ComputeType* moving_variance, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t param_idx = i % param_size;
    const ComputeType inv_variance = rsqrt(moving_variance[param_idx] + epsilon);
    dx[i] = static_cast<T>(static_cast<ComputeType>(dy[i]) * gamma[param_idx] * inv_variance);
  }
}

template<int N>
__global__ void ComputeNhwcPackHalf(const int32_t elem_cnt, const int32_t param_size,
                                    const float epsilon, const Pack<half, N>* dy,
                                    const Pack<float, N>* gamma,
                                    const Pack<float, N>* moving_variance, Pack<half, N>* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    Pack<half, N> dy_pack = dy[i];
    Pack<half, N> dx_pack;
    const int32_t param_idx = i % param_size;
    Pack<float, N> gamma_pack = gamma[param_idx];
    Pack<float, N> moving_variance_pack = moving_variance[param_idx];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float inv_variance = rsqrt(moving_variance_pack.elem[j] + epsilon);
      dx_pack.elem[j] = static_cast<float>(static_cast<float>(dy_pack.elem[j]) * gamma_pack.elem[j]
                                           * inv_variance);
    }
    dx[i] = dx_pack;
  }
}

template<typename T, typename ComputeType>
void DispatchKernel(cudaStream_t cuda_stream, DataType data_type, const int32_t elem_cnt,
                    const int32_t inner_dim_size, const int32_t param_size, const float epsilon,
                    const T* dy, const ComputeType* gamma, const ComputeType* moving_variance,
                    T* dx) {
  if (inner_dim_size == 1) {
    if (param_size % 2 == 0 && data_type == DataType::kFloat16) {
      ComputeNhwcPackHalf<2>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
              elem_cnt / 2, param_size / 2, epsilon, reinterpret_cast<const Pack<half, 2>*>(dy),
              reinterpret_cast<const Pack<float, 2>*>(gamma),
              reinterpret_cast<const Pack<float, 2>*>(moving_variance),
              reinterpret_cast<Pack<half, 2>*>(dx));
    } else {
      ComputeNhwc<T, ComputeType>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
              elem_cnt, param_size, epsilon, dy, gamma, moving_variance, dx);
    }
  } else {
    if (inner_dim_size % 2 == 0 && data_type == DataType::kFloat16) {
      ComputeNchwPackHalf<2>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
              elem_cnt / 2, inner_dim_size, param_size, epsilon,
              reinterpret_cast<const Pack<half, 2>*>(dy), gamma, moving_variance,
              reinterpret_cast<Pack<half, 2>*>(dx));
    } else {
      ComputeNchw<T, ComputeType>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
              elem_cnt, inner_dim_size, param_size, epsilon, dy, gamma, moving_variance, dx);
    }
  }
}

template<typename T, typename ComputeType>
class BnEvalBackwardKernel final : public user_op::OpKernel {
 public:
  BnEvalBackwardKernel() = default;
  ~BnEvalBackwardKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const user_op::Tensor* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t axis = ctx->Attr<int32_t>("axis");
    const float epsilon = ctx->Attr<float>("epsilon");
    CHECK_GE(axis, 0);
    const int64_t inner_dim_size = dy->shape().Count(axis + 1);
    const int64_t param_size = dy->shape().At(axis);
    const int64_t elem_cnt = dy->shape().elem_cnt();
    DispatchKernel(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), dy->data_type(), elem_cnt,
                   inner_dim_size, param_size, epsilon, dy->dptr<T>(), gamma->dptr<ComputeType>(),
                   moving_variance->dptr<ComputeType>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BN_EVAL_BACKWARD_KERNEL(t_type, compute_type)             \
  REGISTER_USER_KERNEL("bn_eval_backward")                                 \
      .SetCreateFn<BnEvalBackwardKernel<t_type, compute_type>>()           \
      .SetIsMatchedHob(                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                  \
          && (user_op::HobDataType("dy", 0) == GetDataType<t_type>::value) \
          && (user_op::HobDataType("moving_variance", 0) == GetDataType<compute_type>::value));

REGISTER_BN_EVAL_BACKWARD_KERNEL(float, float)
REGISTER_BN_EVAL_BACKWARD_KERNEL(half, float)

}  // namespace oneflow
