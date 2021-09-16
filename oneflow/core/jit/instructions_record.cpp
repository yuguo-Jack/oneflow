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

#include <list>
#include "oneflow/core/jit/instructions_record.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/instruction.msg.h"

namespace oneflow {

namespace {

bool* RecordingInstructionsFlag() {
  static thread_local bool recording_instruction = false;
  return &recording_instruction;
}

vm::InstructionMsgList* RecordedInstructionList() {
  static thread_local vm::InstructionMsgList list;
  return &list;
}

}  // namespace

namespace jit {

bool RecordingInstructions() { return *RecordingInstructionsFlag(); }

void StartRecordingInstructions() { *RecordingInstructionsFlag() = true; }

void EndRecordingInstructions() { *RecordingInstructionsFlag() = false; }

void ClearRecordedInstructions() { RecordedInstructionList()->Clear(); }

void RecordInstructions(vm::InstructionMsgList* instruction_msg_list) {
  instruction_msg_list->MoveTo(RecordedInstructionList());
}

void ReplayInstructions() {
  vm::InstructionMsgList instr_msg_list;
  OBJECT_MSG_LIST_FOR_EACH(RecordedInstructionList(), instr_msg) {
    instr_msg_list.EmplaceBack(instr_msg->Clone());
  }
  CHECK_JUST(vm::Run(&instr_msg_list));
}

}  // namespace jit

}  // namespace oneflow
