#ifndef ONEFLOW_CORE_PROFILER_PROFILER_H_
#define ONEFLOW_CORE_PROFILER_PROFILER_H_

#include <string>
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class Kernel;
struct KernelCtx;

namespace profiler {

void TraceKernelForwardDataContentStart(const KernelCtx& ctx, const Kernel* kernel);

void TraceKernelForwardDataContentEnd(const KernelCtx& ctx, const Kernel* kernel);

void RangePush(const std::string& name);

void RangePop();

}  // namespace profiler

}  // namespace oneflow

#define OF_PROFILER_RANGE_PUSH(name) ::oneflow::profiler::RangePush(name)
#define OF_PROFILER_RANGE_POP() ::oneflow::profiler::RangePop()

#endif