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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

namespace {

class NcclSliceBoxingKernelState final : public user_op::OpKernelState {
 public:
  explicit NcclSliceBoxingKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1),
        has_independent_stream_(ctx->op_conf().has_stream_name_hint()),
        stream_name_(""),
        parallel_desc_(ctx->parallel_desc()),
        parallel_id_(ctx->parallel_ctx().parallel_id()) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    if (has_independent_stream_) { stream_name_ = ctx->op_conf().stream_name_hint(); }

    NdSbp src_nd_sbp;
    CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_nd_sbp", &src_nd_sbp));
    NdSbp dst_nd_sbp;
    CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_nd_sbp", &dst_nd_sbp));
    CHECK_EQ(src_nd_sbp.sbp_parallel_size(), parallel_desc_.hierarchy()->NumAxes());
    CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), parallel_desc_.hierarchy()->NumAxes());
    const user_op::TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
    const DataType data_type = in_logical_desc->data_type();
    const DeviceType device_type = parallel_desc_.device_type();
    const Shape& logical_shape = in_logical_desc->shape();
    const int64_t parallel_num = parallel_desc_.parallel_num();
    const std::vector<TensorSliceView>& out_slices =
        GetTensorSliceView(*parallel_desc_.hierarchy(), dst_nd_sbp, logical_shape);
    const std::vector<TensorSliceView>& in_slices =
        GetTensorSliceView(*parallel_desc_.hierarchy(), src_nd_sbp, logical_shape);
    for (int64_t i = 0; i < parallel_num; ++i) {
      const TensorSliceView& cur_rank_in_slice = in_slices.at(parallel_id_);
      const TensorSliceView& in_intersection = out_slices.at(i).Intersect(cur_rank_in_slice);
      if (!in_intersection.IsEmpty()) {
        send_elem_cnts_.push_back(in_intersection.shape().elem_cnt());
        in_tensor_slice_copier_vec_.emplace_back(
            new TensorSliceCopier(in_intersection, cur_rank_in_slice, data_type, device_type));
      } else {
        send_elem_cnts_.push_back(0);
        in_tensor_slice_copier_vec_.emplace_back(nullptr);
      }

      const TensorSliceView& cur_rank_out_slice = out_slices.at(parallel_id_);
      const TensorSliceView& out_interaction = cur_rank_out_slice.Intersect(in_slices.at(i));
      if (!out_interaction.IsEmpty()) {
        recv_elem_cnts_.push_back(out_interaction.shape().elem_cnt());
        out_tensor_slice_copier_vec_.emplace_back(
            new TensorSliceCopier(cur_rank_out_slice, out_interaction, data_type, device_type));
      } else {
        recv_elem_cnts_.push_back(0);
        out_tensor_slice_copier_vec_.emplace_back(nullptr);
      }
    }
  }
  ~NcclSliceBoxingKernelState() {}

  ncclComm_t comm() { return GetOrCreate().comm; }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec() {
    return in_tensor_slice_copier_vec_;
  }
  const std::vector<int64_t>& send_elem_cnts() { return send_elem_cnts_; }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec() {
    return out_tensor_slice_copier_vec_;
  }
  const std::vector<int64_t>& recv_elem_cnts() { return recv_elem_cnts_; }

 private:
  struct Comm {
    Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };

  const Comm& GetOrCreate() {
    if (!comm_) { Init(); }
    return *comm_;
  }

  void Init() {
    std::set<std::pair<int64_t, int64_t>> device_set;
    for (int64_t parallel_id = 0; parallel_id < parallel_desc_.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    ncclComm_t comm;
    if (has_independent_stream_) {
      comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    } else {
      comm = comm_mgr->GetCommForDevice(device_set);
    }
    comm_.reset(new Comm(comm));
  }

  int device_index_;
  bool has_independent_stream_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  int64_t parallel_id_;
  std::unique_ptr<Comm> comm_;

  std::vector<std::shared_ptr<TensorSliceCopier>> in_tensor_slice_copier_vec_;
  std::vector<std::shared_ptr<TensorSliceCopier>> out_tensor_slice_copier_vec_;
  std::vector<int64_t> send_elem_cnts_;
  std::vector<int64_t> recv_elem_cnts_;
};

}  // namespace

class NcclSliceBoxingCopyKernel final : public user_op::OpKernel {
 public:
  NcclSliceBoxingCopyKernel() = default;
  ~NcclSliceBoxingCopyKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclSliceBoxingKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<NcclSliceBoxingKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    ncclComm_t comm = kernel_state->comm();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const std::vector<int64_t>& send_elem_cnts = kernel_state->send_elem_cnts();
    const std::vector<int64_t>& recv_elem_cnts = kernel_state->recv_elem_cnts();
    const int64_t parallel_num = send_elem_cnts.size();
    const DataType data_type = in->data_type();
    std::vector<void*> send_in_ptr;
    std::vector<void*> recv_out_ptr;
    char* tmp_buffer_ptr = tmp_buffer->mut_dptr<char>();
    int64_t offset = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      void* send_ptr = reinterpret_cast<void*>(tmp_buffer_ptr + offset);
      send_in_ptr.push_back(send_ptr);
      offset += send_elem_cnts.at(i) * GetSizeOfDataType(data_type);
    }
    for (int64_t i = 0; i < parallel_num; ++i) {
      void* recv_ptr = reinterpret_cast<void*>(tmp_buffer_ptr + offset);
      recv_out_ptr.push_back(recv_ptr);
      offset += recv_elem_cnts.at(i) * GetSizeOfDataType(data_type);
    }
    CHECK_LE(offset, tmp_buffer->shape().elem_cnt());

    const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec =
        kernel_state->in_tensor_slice_copier_vec();
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (in_tensor_slice_copier_vec.at(i) != nullptr) {
        in_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), send_in_ptr.at(i), in->dptr());
      }
    }

    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    if (ParseBooleanFromEnv("OPTIMIZE_ALL_TO_ALL", false)) {
      OF_NCCL_CHECK(ncclGroupStart());
      for (int64_t i = 0; i < parallel_num; ++i) {
        int send_to = (parallel_id + i) % parallel_num;
        int recv_from = (parallel_id - i + parallel_num) % parallel_num;
        if (send_elem_cnts.at(send_to) != 0) {
          LOG(INFO) << parallel_id << " send " << send_elem_cnts.at(send_to) << " to " << send_to;
          OF_NCCL_CHECK(ncclSend(send_in_ptr.at(send_to), send_elem_cnts.at(send_to),
                                 GetNcclDataType(data_type), send_to, comm, cuda_stream));
        }
        if (recv_elem_cnts.at(recv_from) != 0) {
          LOG(INFO) << parallel_id << " recv " << recv_elem_cnts.at(recv_from) << " from "
                    << recv_from;
          OF_NCCL_CHECK(ncclRecv(recv_out_ptr.at(recv_from), recv_elem_cnts.at(recv_from),
                                 GetNcclDataType(data_type), recv_from, comm, cuda_stream));
        }
      }
      OF_NCCL_CHECK(ncclGroupEnd());
    } else {
      OF_NCCL_CHECK(ncclGroupStart());
      for (int64_t i = 0; i < parallel_num; ++i) {
        if (send_elem_cnts.at(i) != 0) {
          LOG(INFO) << parallel_id << " send " << send_elem_cnts.at(i) << " to " << i;
          OF_NCCL_CHECK(ncclSend(send_in_ptr.at(i), send_elem_cnts.at(i),
                                 GetNcclDataType(data_type), i, comm, cuda_stream));
        }
        if (recv_elem_cnts.at(i) != 0) {
          LOG(INFO) << parallel_id << " recv " << recv_elem_cnts.at(i) << " from " << i;
          OF_NCCL_CHECK(ncclRecv(recv_out_ptr.at(i), recv_elem_cnts.at(i),
                                 GetNcclDataType(data_type), i, comm, cuda_stream));
        }
      }
      OF_NCCL_CHECK(ncclGroupEnd());
    }

    const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec =
        kernel_state->out_tensor_slice_copier_vec();
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i) != nullptr) {
        out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(), recv_out_ptr.at(i));
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  const user_op::TensorDesc& out_tensor = ctx->InputTensorDesc("out", 0);
  const DataType data_type = in_tensor.data_type();
  const size_t buf_bytes =
      (in_tensor.shape().elem_cnt() + out_tensor.shape().elem_cnt()) * GetSizeOfDataType(data_type);
  return buf_bytes;
}

REGISTER_USER_KERNEL("nccl_slice_boxing_copy")
    .SetCreateFn<NcclSliceBoxingCopyKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferTmpBufferSize);

}  // namespace oneflow
