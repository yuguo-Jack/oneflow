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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

/* static */ Maybe<void> NcclSliceBoxingCopyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NcclSliceBoxingCopyOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> NcclSliceBoxingCopyOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  input_nd_sbp->clear_sbp_parallel();
  output_nd_sbp->clear_sbp_parallel();

  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_nd_sbp", input_nd_sbp));
  JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_nd_sbp", output_nd_sbp));

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NcclSliceBoxingCopyOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> NcclSliceBoxingCopyOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn<&SyncLaunched>(ctx);
}

}  // namespace oneflow
