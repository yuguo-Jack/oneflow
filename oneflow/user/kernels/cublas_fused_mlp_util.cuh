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
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>
// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11040

namespace oneflow {

namespace {

constexpr int32_t kAuxReluLdAlignRequirement = 128;

class CublasFusedMLPKernelCache final : public user_op::OpKernelCache {
 public:
  CublasFusedMLPKernelCache() {
    // Just for init.
    OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_a_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_b_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_c_desc, CUDA_R_32F, 1, 1, 1));
  }
  ~CublasFusedMLPKernelCache() override {
    OF_CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_a_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_b_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_c_desc));
  }
  cublasLtMatmulDesc_t operation_desc;
  cublasLtMatrixLayout_t cublas_a_desc;
  cublasLtMatrixLayout_t cublas_b_desc;
  cublasLtMatrixLayout_t cublas_c_desc;
};

std::shared_ptr<CublasFusedMLPKernelCache> CreateCublasFusedMLPKernelCache() {
  std::shared_ptr<CublasFusedMLPKernelCache> cache(new CublasFusedMLPKernelCache());
  return cache;
}

Optional<cudaDataType_t> OptCudaDataType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
    case kBFloat16: return CUDA_R_16BF;
    default: return NullOpt;
  }
}

cudaDataType_t GetCudaDataType(DataType data_type) {
  auto cuda_data_type = OptCudaDataType(data_type);
  CHECK(cuda_data_type.has_value());
  return cuda_data_type.value_or(CUDA_R_32F);
}

cublasComputeType_t GetComputeType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUBLAS_COMPUTE_32F;
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: return CUBLAS_COMPUTE_32F;
    case kBFloat16: return CUBLAS_COMPUTE_32F;
    default: UNIMPLEMENTED(); return CUBLAS_COMPUTE_32F;
  }
}

union CublasScalarParameter {
  double d;
  float s;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cublasComputeType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUBLAS_COMPUTE_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUBLAS_COMPUTE_32F) {
    sp.s = scalar.Value<float>();
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

void InferMatmulCublasMNK(const DimVector& a_shape, const DimVector& b_shape,
                          ep::primitive::BlasTransposeType transpose_a,
                          ep::primitive::BlasTransposeType transpose_b, size_t* cublas_m,
                          size_t* cublas_n, size_t* cublas_k, int64_t* cublas_lda,
                          int64_t* cublas_ldb, int64_t* cublas_ldc) {
  const int64_t num_a_axes = a_shape.size();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.size();
  CHECK_GE(num_b_axes, 2);
  size_t m = 0, n = 0, k = 0;
  if (transpose_a == ep::primitive::BlasTransposeType::N) {
    m = a_shape.at(num_a_axes - 2);
    k = a_shape.at(num_a_axes - 1);
    *cublas_ldb = k;
  } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
    m = a_shape.at(num_a_axes - 1);
    k = a_shape.at(num_a_axes - 2);
    *cublas_ldb = m;
  } else {
    UNIMPLEMENTED();
  }
  if (transpose_b == ep::primitive::BlasTransposeType::N) {
    CHECK_EQ(b_shape.at(num_b_axes - 2), k);
    n = b_shape.at(num_b_axes - 1);
    *cublas_lda = n;
  } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
    CHECK_EQ(b_shape.at(num_b_axes - 1), k);
    n = b_shape.at(num_b_axes - 2);
    *cublas_lda = k;
  } else {
    UNIMPLEMENTED();
  }
  *cublas_m = n;
  *cublas_n = m;
  *cublas_k = k;
  *cublas_ldc = n;
}

void SetCublasMatrixLayout(cublasLtMatrixLayout_t layout_desc, cudaDataType_t cuda_data_type,
                           cublasOperation_t cublas_trans, const size_t cublas_m1,
                           const size_t cublas_n1, int64_t cublas_ld) {
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                                   &cuda_data_type, sizeof(cuda_data_type)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_ROWS,
      cublas_trans == CUBLAS_OP_N ? &cublas_m1 : &cublas_n1, sizeof(cublas_m1)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_COLS,
      cublas_trans == CUBLAS_OP_N ? &cublas_n1 : &cublas_m1, sizeof(cublas_m1)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_LD,
                                                   &cublas_ld, sizeof(cublas_ld)));
}

void SetCublasEpilogue(const CublasFusedMLPKernelCache* matmul_cache, cublasLtEpilogue_t epilogue,
                       const void* bias_ptr, const void* aux_ptr) {
  if (epilogue == CUBLASLT_EPILOGUE_RELU_BIAS || epilogue == CUBLASLT_EPILOGUE_BIAS
      || epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS || epilogue == CUBLASLT_EPILOGUE_DRELU_BGRAD) {
    // Set epilogue
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // Set bias ptr
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
                                                   sizeof(bias_ptr)));
  } else {
    Error::UnimplementedError() << "Unsupported Epilogue. ";
  }

  // TODO: Support GELU_AUX_BIAS
  if (epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS || epilogue == CUBLASLT_EPILOGUE_DRELU_BGRAD) {
    // Set aux ptr for backward.
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                   &aux_ptr, sizeof(aux_ptr)));
  } else {
    // Clear Aux ptr.
    aux_ptr = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                   &aux_ptr, sizeof(aux_ptr)));
  }
}

long AlignReluAuxLd(long aux_ld) {
  /*
  ReLu bit-mask matrix leading dimension in elements.
  Must be divisible by 128 and be no less than the number of rows in the output matrix.
  */
  long old_aux_ld = aux_ld;
  return ((old_aux_ld + kAuxReluLdAlignRequirement - 1) / kAuxReluLdAlignRequirement)
         * kAuxReluLdAlignRequirement;
}

void SetCublasAttr(const CublasFusedMLPKernelCache* matmul_grad_cache,
                   const cublasComputeType_t cublas_compute_dtype,
                   const cudaDataType_t cuda_data_type, bool need_aux,
                   ep::primitive::BlasTransposeType transpose_a,
                   ep::primitive::BlasTransposeType transpose_b, cublasLtEpilogue_t epilogue,
                   const void* d_bias_ptr, const void* aux_ptr, size_t cublas_m, size_t cublas_n,
                   size_t cublas_k, int64_t cublas_lda, int64_t cublas_ldb, int64_t cublas_ldc) {
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &cublas_compute_dtype,
      sizeof(cublas_compute_dtype)));

  // For best performance when using the bias vector, specify beta == 0 and
  // CUBLASLT_POINTER_MODE_HOST.(from
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtPointerMode_t)
  cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));

  // transpose_a = False, transpose_b = True. But in cublas is reversed.
  const cublasOperation_t cublas_trans_a =
      transpose_b == ep::primitive::BlasTransposeType::T ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b =
      transpose_a == ep::primitive::BlasTransposeType::T ? CUBLAS_OP_T : CUBLAS_OP_N;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
                                                 CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a,
                                                 sizeof(cublas_trans_a)));
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
                                                 CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b,
                                                 sizeof(cublas_trans_b)));

  // Set epilogue
  SetCublasEpilogue(matmul_grad_cache, epilogue, d_bias_ptr, aux_ptr);
  /*
  Set AUX pointer LD
  If is used for CUBLASLT_EPILOGUE_DRELU_BGRAD, the AUX_LD need to align 128bit.
  If is used for CUBLASLT_EPILOGUE_DGELU_BGRAD, the AUX_LD need to align 8.
  For more details you can refer to CUBLAS docs:
  https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t
  `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`.
  */
  if (need_aux) {
    long aligned_aux_ld = AlignReluAuxLd(cublas_ldc);
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                   &aligned_aux_ld, sizeof(aligned_aux_ld)));
  }
  // Set matrix layout
  SetCublasMatrixLayout(matmul_grad_cache->cublas_a_desc, cuda_data_type, cublas_trans_a, cublas_m,
                        cublas_k, cublas_lda);
  SetCublasMatrixLayout(matmul_grad_cache->cublas_b_desc, cuda_data_type, cublas_trans_b, cublas_k,
                        cublas_n, cublas_ldb);
  SetCublasMatrixLayout(matmul_grad_cache->cublas_c_desc, cuda_data_type, CUBLAS_OP_N, cublas_m,
                        cublas_n, cublas_ldc);
}

}  // namespace

}  // namespace oneflow

#endif
