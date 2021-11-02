#include "oneflow/core/profiler/profiler.h"

#include <nvtx3/nvToolsExt.h>

namespace oneflow {

namespace profiler {

void RangePush(const std::string& name) { nvtxRangePushA(name.c_str()); }

void RangePop() { nvtxRangePop(); }

void TraceKernelForwardDataContentStart(const KernelCtx& ctx, const Kernel* kernel) {
  OF_PROFILER_RANGE_PUSH(kernel->op_conf().name());
}

void TraceKernelForwardDataContentEnd(const KernelCtx& ctx, const Kernel* kernel) {
  OF_PROFILER_RANGE_POP();
}

}
}
