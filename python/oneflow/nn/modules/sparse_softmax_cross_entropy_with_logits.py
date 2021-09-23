"""
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
"""
import oneflow as flow

def distributed_sparse_softmax_cross_entropy_with_logits(labels, logits):
    (_, out) = flow._C.distributed_sparse_softmax_cross_entropy(logits, labels)
    return out

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
