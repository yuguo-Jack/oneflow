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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/add.h"
namespace oneflow {

namespace {

bool NdSbpNoPartialParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  FOR_RANGE(int64_t, i, 0, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) { return false; }
  }
  return true;
}

// Go through all the ranks while transfer between two nd sbps with no PartialSum under the same
// placement.
// NOTE: We need to make sure no partial sums in the sbps of the producer and consumer.
void DfsTraverseRanks4NdSbp(
    int32_t depth, std::vector<int64_t>& in_parallel_ids,
    const std::vector<int64_t>& out_parallel_ids, const Shape& parallel_hierarchy,
    const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>& hierarchy_index_helper,
    const NdSbp& in_nd_sbp, const std::function<void(int32_t, int32_t)>& visit) {
  if (depth >= parallel_hierarchy.NumAxes()) {
    visit(hierarchy_index_helper.NdIndexToOffset(out_parallel_ids.data(),
                                                 parallel_hierarchy.NumAxes()),
          hierarchy_index_helper.NdIndexToOffset(in_parallel_ids.data(),
                                                 parallel_hierarchy.NumAxes()));
    return;
  }
  if (in_nd_sbp.sbp_parallel(depth).has_broadcast_parallel()) {
    // If Broadcast in the sbp of the producer, only visit those ranks with the same id as the
    // current rank along the depth-dimension.
    in_parallel_ids[depth] = out_parallel_ids[depth];
    DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, parallel_hierarchy,
                           hierarchy_index_helper, in_nd_sbp, visit);
  } else {
    // If Split or PartialSum, go through all the ranks along the depth-dimension.
    for (int64_t i = 0; i < parallel_hierarchy.dim_vec().at(depth); i++) {
      in_parallel_ids[depth] = i;
      DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, parallel_hierarchy,
                             hierarchy_index_helper, in_nd_sbp, visit);
    }
  }
}

void DfsTraverse4NdSbp(int64_t out_id, const std::shared_ptr<Shape> parallel_hierarchy,
                       const NdSbp& in_nd_sbp, const std::function<void(int32_t, int32_t)>& visit) {
  int32_t hierarchy_dimension = parallel_hierarchy->NumAxes();
  const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> hierarchy_index_helper(
      parallel_hierarchy->dim_vec().data(), hierarchy_dimension);
  std::vector<int64_t> in_parallel_ids(hierarchy_dimension);
  std::vector<int64_t> out_parallel_ids(hierarchy_dimension);
  hierarchy_index_helper.OffsetToNdIndex(out_id, out_parallel_ids.data(), hierarchy_dimension);
  DfsTraverseRanks4NdSbp(0, in_parallel_ids, out_parallel_ids, *parallel_hierarchy,
                         hierarchy_index_helper, in_nd_sbp, visit);
}
}  // namespace

class NcclSendRecvBoxingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSendRecvBoxingKernel);
  NcclSendRecvBoxingKernel() = default;
  ~NcclSendRecvBoxingKernel() override = default;

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec() const {
    return in_tensor_slice_copier_vec_;
  }
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec() const {
    return out_tensor_slice_copier_vec_;
  }
  const std::vector<int64_t>& send_elem_cnts() const { return send_elem_cnts_; }
  const std::vector<int64_t>& recv_elem_cnts() const { return recv_elem_cnts_; }
  ncclComm_t comm() const { return GetOrCreate().comm; }

 private:
  struct Comm {
    Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };

  void Init() const {
    ParallelDesc parallel_desc(parallel_conf_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
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

  const Comm& GetOrCreate() const {
    if (!comm_) { Init(); }
    return *comm_;
  }

  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;

  bool has_independent_stream_;
  std::string stream_name_;
  ParallelConf parallel_conf_;
  mutable std::unique_ptr<Comm> comm_;
  bool src_nd_sbp_no_partial_parallel_;
  std::vector<std::shared_ptr<TensorSliceCopier>> in_tensor_slice_copier_vec_;
  std::vector<std::shared_ptr<TensorSliceCopier>> out_tensor_slice_copier_vec_;
  std::vector<int64_t> send_elem_cnts_;
  std::vector<int64_t> recv_elem_cnts_;
};

void NcclSendRecvBoxingKernel::ForwardDataContent(KernelContext* ctx) const {
  const Blob* in = ctx->BnInOp2Blob("in");
  Blob* out = ctx->BnInOp2Blob("out");
  Blob* buf = ctx->BnInOp2Blob("buf");

  ncclComm_t comm = this->comm();
  cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
  const std::vector<int64_t>& send_elem_cnts = this->send_elem_cnts();
  const std::vector<int64_t>& recv_elem_cnts = this->recv_elem_cnts();
  const int64_t parallel_num = send_elem_cnts.size();
  const DataType data_type = in->data_type();
  std::vector<void*> send_in_ptr;
  std::vector<void*> recv_out_ptr;
  char* buf_ptr = buf->mut_dptr<char>();
  int64_t offset = 0;
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* send_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    send_in_ptr.push_back(send_ptr);
    offset += send_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* recv_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    recv_out_ptr.push_back(recv_ptr);
    offset += recv_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec =
      this->in_tensor_slice_copier_vec();
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (in_tensor_slice_copier_vec.at(i)) {
      in_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), send_in_ptr.at(i), in->dptr());
    }
  }
  const int64_t parallel_id = this->kernel_conf().parallel_ctx().parallel_id();
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (send_elem_cnts.at(i) != 0) {
      LOG(INFO) << parallel_id << " send " << send_elem_cnts.at(i) << " to " << i;
      OF_NCCL_CHECK(ncclSend(send_in_ptr.at(i), send_elem_cnts.at(i), GetNcclDataType(data_type), i,
                             comm, cuda_stream));
    }
    if (recv_elem_cnts.at(i) != 0) {
      LOG(INFO) << parallel_id << " recv " << recv_elem_cnts.at(i) << " from " << i;
      OF_NCCL_CHECK(ncclRecv(recv_out_ptr.at(i), recv_elem_cnts.at(i), GetNcclDataType(data_type),
                             i, comm, cuda_stream));
    }
  }
  OF_NCCL_CHECK(ncclGroupEnd());
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec =
      this->out_tensor_slice_copier_vec();

  if (src_nd_sbp_no_partial_parallel_) {
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(), recv_out_ptr.at(i));
      }
    }
  } else {
    std::unique_ptr<ep::primitive::Add> primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->stream()->device_type(),
                                                               out->data_type());
    CHECK(primitive);
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
    CHECK(memset_primitive);
    bool is_first_slice = true;
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        if (is_first_slice) {
          is_first_slice = false;
          if (recv_elem_cnts.at(i) != out->shape().elem_cnt()) {
            // if not same shape, memset out
            memset_primitive->Launch(ctx->stream(), out->mut_dptr(), 0,
                                     out->shape().elem_cnt() * GetSizeOfDataType(data_type));
          }
          out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(),
                                                  recv_out_ptr.at(i));
        } else {
          if (recv_elem_cnts.at(i) == out->shape().elem_cnt()) {
            primitive->Launch(ctx->stream(), out->dptr(), recv_out_ptr.at(i), out->mut_dptr(),
                              out->shape().elem_cnt());
          } else {
            void* out_buf = reinterpret_cast<void*>(buf_ptr + offset);
            memset_primitive->Launch(ctx->stream(), out_buf, 0,
                                     out->shape().elem_cnt() * GetSizeOfDataType(data_type));
            out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out_buf, recv_out_ptr.at(i));
            primitive->Launch(ctx->stream(), out->dptr(), out_buf, out->mut_dptr(),
                              out->shape().elem_cnt());
          }
        }
      }
    }
  }
}

void NcclSendRecvBoxingKernel::VirtualKernelInit(KernelContext* ctx) {
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  has_independent_stream_ = this->op_conf().has_stream_name_hint();
  if (has_independent_stream_) { stream_name_ = this->op_conf().stream_name_hint(); }
  parallel_conf_ = conf.parallel_conf();
  const int64_t parallel_id = this->kernel_conf().parallel_ctx().parallel_id();
  ParallelDesc parallel_desc(parallel_conf_);
  const NdSbp& src_nd_sbp = conf.src_nd_sbp();
  const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  src_nd_sbp_no_partial_parallel_ = NdSbpNoPartialParallel(src_nd_sbp);
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), parallel_hierarchy->NumAxes());
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), parallel_hierarchy->NumAxes());
  const DataType data_type = this->kernel_conf().data_type();
  const DeviceType device_type = parallel_desc.device_type();
  const Shape& logical_shape = Shape(conf.logical_shape());
  const int64_t parallel_num = parallel_desc.parallel_num();
  const std::vector<TensorSliceView>& out_slices =
      GetTensorSliceView(*parallel_desc.hierarchy(), dst_nd_sbp, logical_shape);
  const std::vector<TensorSliceView>& in_slices =
      GetTensorSliceView(*parallel_desc.hierarchy(), src_nd_sbp, logical_shape);

  // add to cur_out_slice
  recv_elem_cnts_.resize(parallel_num);
  out_tensor_slice_copier_vec_.resize(parallel_num);
  const TensorSliceView& cur_rank_out_slice = out_slices.at(parallel_id);
  const auto& add_to_out_slice_vec = [&](int32_t out_id, int32_t in_id) {
    CHECK_EQ(out_id, parallel_id);
    const TensorSliceView& in_slice = in_slices.at(in_id);
    const TensorSliceView& intersection = cur_rank_out_slice.Intersect(in_slice);
    if (intersection.IsEmpty()) { return; }
    recv_elem_cnts_.at(in_id) = intersection.shape().elem_cnt();
    out_tensor_slice_copier_vec_.at(in_id).reset(
        new TensorSliceCopier(cur_rank_out_slice, intersection, data_type, device_type));
  };
  DfsTraverse4NdSbp(parallel_id, parallel_hierarchy, src_nd_sbp, add_to_out_slice_vec);

  // add to cur_in_slice
  send_elem_cnts_.resize(parallel_num);
  in_tensor_slice_copier_vec_.resize(parallel_num);
  const TensorSliceView& cur_rank_in_slice = in_slices.at(parallel_id);
  const auto& add_to_in_slice_vec = [&](int32_t out_id, int32_t in_id) {
    if (in_id != parallel_id) { return; }
    const TensorSliceView& out_slice = out_slices.at(out_id);
    const TensorSliceView& intersection = out_slice.Intersect(cur_rank_in_slice);
    if (intersection.IsEmpty()) { return; }
    send_elem_cnts_.at(out_id) = intersection.shape().elem_cnt();
    in_tensor_slice_copier_vec_.at(out_id).reset(
        new TensorSliceCopier(intersection, cur_rank_in_slice, data_type, device_type));
  };
  for (int64_t i = 0; i < parallel_num; ++i) {
    DfsTraverse4NdSbp(i, parallel_hierarchy, src_nd_sbp, add_to_in_slice_vec);
  }
}

REGISTER_KERNEL(OperatorConf::kNcclSendRecvBoxingConf, NcclSendRecvBoxingKernel);

}  // namespace oneflow
