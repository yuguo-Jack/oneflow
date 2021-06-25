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

import numpy as np
import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op


class AsTensor(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dtype, device):
        if isinstance(x, flow.Tensor) and x.dtype == dtype and x.device == device:
            return x

        if dtype == None:
            if isinstance(x, flow.Tensor):
                dtype = x.dtype
            else:
                raise NotImplementedError("Can not infer datatype in this case!")
        
        if device == None:
            if isinstance(x, flow.Tensor):
                device = x.device
            else:
                raise NotImplementedError("Can not infer device_type in this case!")

        if isinstance(x, int) or isinstance(x, float):
            x = flow.Tensor(
                [float(x)],
                dtype=dtype,
                device=device,
            )
        
        if isinstance(x, np.ndarry):
            x = flow.Tensor(x, dtype=dtype, device=device)
        
        if isinstance(x, list) or isinstance(x, tuple):
            x = flow.Tensor(np.array(x), dtype=dtype, device=device)
        
        if isinstance(x, flow.Tensor):
            x.dtype = dtype
            return x.to(device)

        return x

@oneflow_export("as_tensor")
@register_tensor_op("as_tensor")
@experimental_api
def astensor_op(x, dtype = None, device = None):
    r"""
    """
    return AsTensor()(x, dtype, device)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
