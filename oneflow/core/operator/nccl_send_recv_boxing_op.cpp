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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

bool NdSbpHasPartialParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  FOR_RANGE(int64_t, i, 0, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) { return true; }
  }
  return false;
}

bool NdSbpHasBroadcastParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  FOR_RANGE(int64_t, i, 0, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i).has_broadcast_parallel()) { return true; }
  }
  return false;
}

}  // namespace

class NcclSendRecvBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSendRecvBoxingOp);
  NcclSendRecvBoxingOp() = default;
  ~NcclSendRecvBoxingOp() override = default;

  Maybe<void> InitFromOpConf() override;
  Maybe<void> InferInternalBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const JobDesc* job_desc) const override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    UNIMPLEMENTED_THEN_RETURN();
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

Maybe<void> NcclSendRecvBoxingOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  EnrollTmpBn("buf");
  return Maybe<void>::Ok();
}

Maybe<void> NcclSendRecvBoxingOp::InferInternalBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* buf = GetBlobDesc4BnInOp("buf");
  buf->set_data_type(in->data_type());
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  const NdSbp& src_nd_sbp = conf.src_nd_sbp();
  const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
  int64_t in_buffer_cnt = in->shape().elem_cnt();
  int64_t out_buffer_cnt = out->shape().elem_cnt();
  const int64_t parallel_num = parallel_ctx->parallel_num();
  // TODO: exactly calculate buf size , now is the max value
  if (NdSbpHasPartialParallel(src_nd_sbp)) { out_buffer_cnt *= (parallel_num + 1); }
  if (NdSbpHasBroadcastParallel(dst_nd_sbp)) { in_buffer_cnt *= parallel_num; }
  buf->mut_shape() = Shape({in_buffer_cnt + out_buffer_cnt});
  return Maybe<void>::Ok();
}

LogicalBlobId NcclSendRecvBoxingOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().nccl_send_recv_boxing_conf().lbi();
}

LogicalBlobId NcclSendRecvBoxingOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().nccl_send_recv_boxing_conf().lbi();
}

Maybe<void> NcclSendRecvBoxingOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  const NdSbp& src_nd_sbp = conf.src_nd_sbp();
  const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
  const ParallelDesc& parallel_desc = ParallelDesc(conf.parallel_conf());
  Shape logical_shape(conf.logical_shape());
  *out_blob_desc = *in_blob_desc;
  std::shared_ptr<Shape> in_shape =
      JUST(GetPhysicalShape(logical_shape, src_nd_sbp, parallel_desc, 0));
  CHECK_EQ_OR_RETURN(*in_shape, in_blob_desc->shape());
  std::shared_ptr<Shape> out_shape =
      JUST(GetPhysicalShape(logical_shape, dst_nd_sbp, parallel_desc, 0));
  out_blob_desc->mut_shape() = *out_shape;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kNcclSendRecvBoxingConf, NcclSendRecvBoxingOp);

}  // namespace oneflow
