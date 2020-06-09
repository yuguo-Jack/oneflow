#ifndef ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class RandomShuffleDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  RandomShuffleDataset(user_op::KernelInitContext* ctx,
                       std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : loader_(std::move(data_set)) {
    // random
    seed_ = ctx->Attr<int64_t>("seed");
    if (seed_ == -1) { seed_ = NewRandomSeed(); }
    std::seed_seq seq({seed_});
    rand_engine_ = std::default_random_engine(seq);

    // fill buffer
    initial_buffer_fill_ = ctx->Attr<int32_t>("shuffle_buffer_size");
    int32_t remain_cnt = initial_buffer_fill_;
    while (remain_cnt > 0) {
      LoadTargetPtrList sample_list = loader_->Next();
      for (auto& sample_ptr : sample_list) {
        sample_buffer_.push_back(std::move(sample_ptr));
        remain_cnt--;
      }
    }
  }
  ~RandomShuffleDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret = loader_->Next();
    for (auto& sample_ptr : ret) {
      std::uniform_int_distribution<> dis(0, sample_buffer_.size() - 1);
      int offset = dis(rand_engine_);
      std::swap(sample_buffer_[offset], sample_ptr);
    }
    return ret;
  }

 private:
  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::vector<LoadTargetPtr> sample_buffer_;

  int32_t initial_buffer_fill_;

  std::default_random_engine rand_engine_;
  int64_t seed_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
