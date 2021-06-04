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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class LogSigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log_sigmoid").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("log_sigmoid")
@register_tensor_op("log_sigmoid")
@experimental_api
def log_sigmoid_op(input):
    r"""Applies the element-wise function:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log_sigmoid(x).numpy()
        >>> out
        array([-0.24100842, -0.20141333, -0.0650436 ], dtype=float32)

    """
    return LogSigmoid()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
