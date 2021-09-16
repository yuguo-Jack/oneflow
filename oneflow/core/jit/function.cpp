#include "oneflow/core/jit/function.h"

namespace oneflow {
namespace jit {

class FunctionCacheKey final {
 public:

  static Maybe<FunctionCacheKey> New(const TensorTuple& inputs) {
    TODO();
  }

  bool operator==(const FunctionCacheKey& other) const {
    TODO();
  }
  size_t CalcHashValue() const {
    TODO();
  }

 private:
  FunctionCacheKey(const std::shared_ptr<std::vector<std::shared_ptr<one::TensorMeta>>>& input_tensor_metas,
                   bool is_autograd_enabled)
      : input_tensor_metas_(input_tensor_metas), is_autograd_enabled_(is_autograd_enabled) {}

  std::shared_ptr<std::vector<one::TensorMetaHashKey>> input_tensor_metas_;
  // TODO(lixinqi): input_autograd_metas_;
  bool is_autograd_enabled_;
};

}
}

namespace std {

template<>
struct hash<oneflow::jit::FunctionCacheKey> final {
  size_t operator()(const oneflow::jit::FunctionCacheKey& key) const { return key.CalcHashValue(); }
};

}

namespace oneflow {
namespace jit {

Maybe<Function> Function::FindOrCreate(const TensorTuple& inputs) {
  TODO();
}

Maybe<TensorTuple> Function::operator()(const TensorTuple& inputs) const {
  TODO();
}

}
}
