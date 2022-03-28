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
import warnings
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_inplace_valid


class LeakyReLU(Module):
    """Applies the element-wise function:

    .. math::
        \\text{LeakyRELU}(x) = \\begin{cases}
            x, & \\text{ if } x \\geq 0 \\\\
            \\text{negative_slope} \\times x, & \\text{ otherwise }
        \\end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.LeakyReLU(0.1)
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 3.0000, 4.0000], dtype=oneflow.float32)
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("LeakyReLU module do not support inplace now")
        return flow._C.leaky_relu(x, alpha=self.negative_slope)

    def extra_repr(self):
        param_str = f"negative_slope={self.negative_slope}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
