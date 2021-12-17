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
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/event_phy_instr_operand.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"

namespace oneflow {
namespace vm {

template<typename CudaStreamT>
class CudaStreamEventRecordInstructionType : public vm::InstructionType {
 public:
  CudaStreamEventRecordInstructionType() = default;
  ~CudaStreamEventRecordInstructionType() override = default;

  using stream_type = CudaStreamT;

  void InitInstructionState(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* event_provider = dynamic_cast<QueryCudaEventProvider*>(stream->device_ctx().get());
    const auto& cuda_event = CHECK_NOTNULL(event_provider)->GetCudaEvent();
    auto* operand =
        static_cast<EventPhyInstrOperand*>(instruction->instr_msg().phy_instr_operand().get());
    *CHECK_NOTNULL(operand)->mut_event() = cuda_event;
  }
  void Compute(vm::Instruction* instruction) const override {
    auto* device_ctx = instruction->mut_stream()->device_ctx().get();
    auto* operand =
        static_cast<EventPhyInstrOperand*>(instruction->instr_msg().phy_instr_operand().get());
    CHECK_NOTNULL(operand);
    auto* cuda_event = static_cast<CudaEvent*>(operand->event().get());
    CHECK_NOTNULL(cuda_event);
    cudaSetDevice(cuda_event->device_id());
    OF_CUDA_CHECK(cudaEventRecord(*cuda_event->mut_event(), device_ctx->cuda_stream()));
  }

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
};

COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<CudaStreamType>>(
    "cuda.StreamEventRecord"));
COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<CudaStreamType>>(
    "gpu.StreamEventRecord"));
COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<CudaStreamType>>(
    "sync_launched_nccl.StreamEventRecord"));
COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<AsyncCudaStreamType>>(
    "async_launched_nccl.StreamEventRecord"));
COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<CudaCopyH2DStreamType>>(
    "cuda_h2d.StreamEventRecord"));
COMMAND(vm::RegisterInstructionType<CudaStreamEventRecordInstructionType<CudaCopyD2HStreamType>>(
    "cuda_d2h.StreamEventRecord"));
}  // namespace vm
}  // namespace oneflow
