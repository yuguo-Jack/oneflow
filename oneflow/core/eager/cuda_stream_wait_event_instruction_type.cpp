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
class CudaStreamWaitEventInstructionType : public vm::InstructionType {
 public:
  CudaStreamWaitEventInstructionType() = default;
  ~CudaStreamWaitEventInstructionType() override = default;

  using stream_type = CudaStreamT;

  void Compute(vm::Instruction* instruction) const override {
    auto* device_ctx = instruction->mut_stream()->device_ctx().get();
    auto* operand =
        static_cast<EventPhyInstrOperand*>(instruction->instr_msg().phy_instr_operand().get());
    CHECK_NOTNULL(operand);
    auto* cuda_event = static_cast<CudaEvent*>(operand->event().get());
    CHECK_NOTNULL(cuda_event);
    cudaSetDevice(cuda_event->device_id());
    OF_CUDA_CHECK(cudaStreamWaitEvent(device_ctx->cuda_stream(), *cuda_event->mut_event(),
                                      cudaEventWaitDefault));
  }
  void DeleteInstructionState(Instruction* instruction) const override {
    auto* operand =
        static_cast<EventPhyInstrOperand*>(instruction->instr_msg().phy_instr_operand().get());
    CHECK_NOTNULL(operand)->mut_event()->reset();
  }
  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
};

COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<CudaStreamType>>(
    "cuda.StreamWaitEvent"));
COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<CudaStreamType>>(
    "gpu.StreamWaitEvent"));
COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<CudaStreamType>>(
    "sync_launched_nccl.StreamWaitEvent"));
COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<AsyncCudaStreamType>>(
    "async_launched_nccl.StreamWaitEvent"));
COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<CudaCopyH2DStreamType>>(
    "cuda_h2d.StreamWaitEvent"));
COMMAND(vm::RegisterInstructionType<CudaStreamWaitEventInstructionType<CudaCopyD2HStreamType>>(
    "cuda_d2h.StreamWaitEvent"));
}  // namespace vm
}  // namespace oneflow
