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

import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_embedding_impl(test_case, device):
    weight = np.array(
        [
            [0.68258786, 0.6957856, 1.1829041],
            [1.0154, -1.0616943, 0.50303376],
            [0.29679507, 0.65562993, 1.0424724],
            [-0.42980736, -0.35347632, -0.15600166],
            [0.6763601, -0.24286619, -2.0873115],
            [-0.13371214, -0.5589277, 1.9173933],
            [0.08762296, 1.0264007, -0.67938024],
            [0.32019204, -0.26137325, -1.3534237],
            [-1.1555519, -0.67776406, 0.27372134],
            [1.0615997, -0.59715784, 1.9855849],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [1.0154, -1.0616943, 0.50303376],
                [0.29679507, 0.65562993, 1.0424724],
                [0.6763601, -0.24286619, -2.0873115],
                [-0.13371214, -0.5589277, 1.9173933],
            ],
            [
                [0.6763601, -0.24286619, -2.0873115],
                [-0.42980736, -0.35347632, -0.15600166],
                [0.29679507, 0.65562993, 1.0424724],
                [1.0615997, -0.59715784, 1.9855849],
            ],
        ],
        dtype=np.float32,
    )
    indices = flow.tensor(
        [[1, 2, 4, 5], [4, 3, 2, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    m = flow.nn.Embedding(10, 3, _weight=flow.Tensor(weight))
    m = m.to(device)
    y = m(indices)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))
    y = y.sum()
    y.backward()
    weight_grad_np = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    test_case.assertTrue(
        np.allclose(m.weight.grad.numpy(), weight_grad_np, 1e-05, 1e-05)
    )


def _test_embedding_renorm(test_case, device):
    embedding = flow.nn.Embedding(10, 3, max_norm=1.0).to(device)
    indices = flow.tensor(
        [[1, 2, 4, 5], [4, 3, 2, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    output = embedding(indices)
    test_case.assertTrue(output.data.norm(p=2, dim=2).le(1).all())


def _test_embedding_padding_idx(test_case, device):
    indices = flow.tensor(
        [[1, 0, 4, 8], [8, 3, 0, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    embedding = flow.nn.Embedding(10, 3, padding_idx=0).to(device)
    output = embedding(indices)
    test_case.assertEqual(output[0][1].sum(), 0)
    test_case.assertEqual(output[1][2].sum(), 0)

    # negative indexing check for padding_idx
    # padding_idx=-2, num_embeddings=10 ==> index 8 padded
    embedding = flow.nn.Embedding(10, 3, padding_idx=-2).to(device)
    output = embedding(indices)
    test_case.assertEqual(output[0][3].sum(), 0)
    test_case.assertEqual(output[1][0].sum(), 0)

    # out of bounds check for padding_idx
    test_case.assertRaises(
        AssertionError,
        flow.nn.Embedding,
        num_embeddings=10,
        embedding_dim=3,
        padding_idx=25,
    )
    test_case.assertRaises(
        AssertionError,
        flow.nn.Embedding,
        num_embeddings=10,
        embedding_dim=3,
        padding_idx=-25,
    )

    padding_idx = 0
    embedding = flow.nn.Embedding(10, 3, padding_idx=padding_idx).to(device)
    indices = flow.tensor(
        [[1, 0, 4, 8], [8, 3, 0, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    pre = embedding.weight[padding_idx].clone()
    embedding(indices).sum().backward()
    after = (embedding.weight + embedding.weight.grad)[padding_idx]
    embedding.zero_grad()
    test_case.assertTrue(flow.equal(after, pre).all())


def _test_embedding_scale_by_freq(test_case, device):
    weight = np.array(
        [
            [0.68258786, 0.6957856, 1.1829041],
            [1.0154, -1.0616943, 0.50303376],
            [0.29679507, 0.65562993, 1.0424724],
            [-0.42980736, -0.35347632, -0.15600166],
            [0.6763601, -0.24286619, -2.0873115],
            [-0.13371214, -0.5589277, 1.9173933],
            [0.08762296, 1.0264007, -0.67938024],
            [0.32019204, -0.26137325, -1.3534237],
            [-1.1555519, -0.67776406, 0.27372134],
            [1.0615997, -0.59715784, 1.9855849],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [1.0154, -1.0616943, 0.50303376],
                [0.29679507, 0.65562993, 1.0424724],
                [0.6763601, -0.24286619, -2.0873115],
                [-0.13371214, -0.5589277, 1.9173933],
            ],
            [
                [0.6763601, -0.24286619, -2.0873115],
                [-0.42980736, -0.35347632, -0.15600166],
                [0.29679507, 0.65562993, 1.0424724],
                [1.0615997, -0.59715784, 1.9855849],
            ],
        ],
        dtype=np.float32,
    )
    indices = flow.tensor(
        [[1, 2, 4, 5], [4, 3, 2, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    m = flow.nn.Embedding(10, 3, scale_grad_by_freq=True, _weight=flow.Tensor(weight))
    m = m.to(device)
    y = m(indices)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))
    y = y.sum()
    y.backward()
    weight_grad_np = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    test_case.assertTrue(
        np.allclose(m.weight.grad.numpy(), weight_grad_np, 1e-05, 1e-05)
    )


def _test_embedding_functional_impl(test_case, device):
    weight_ = np.array(
        [
            [0.68258786, 0.6957856, 1.1829041],
            [1.0154, -1.0616943, 0.50303376],
            [0.29679507, 0.65562993, 1.0424724],
            [-0.42980736, -0.35347632, -0.15600166],
            [0.6763601, -0.24286619, -2.0873115],
            [-0.13371214, -0.5589277, 1.9173933],
            [0.08762296, 1.0264007, -0.67938024],
            [0.32019204, -0.26137325, -1.3534237],
            [-1.1555519, -0.67776406, 0.27372134],
            [1.0615997, -0.59715784, 1.9855849],
        ],
        dtype=np.float32,
    )
    weight = flow.Tensor(weight_)
    weight = weight.to(device)
    weight.requires_grad = True
    output = np.array(
        [
            [
                [1.0154, -1.0616943, 0.50303376],
                [0.29679507, 0.65562993, 1.0424724],
                [0.6763601, -0.24286619, -2.0873115],
                [-0.13371214, -0.5589277, 1.9173933],
            ],
            [
                [0.6763601, -0.24286619, -2.0873115],
                [-0.42980736, -0.35347632, -0.15600166],
                [0.29679507, 0.65562993, 1.0424724],
                [1.0615997, -0.59715784, 1.9855849],
            ],
        ],
        dtype=np.float32,
    )
    indices = flow.tensor(
        [[1, 2, 4, 5], [4, 3, 2, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    y = flow.nn.functional.embedding(indices, weight)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))
    y = y.sum()
    y.backward()
    weight_grad_np = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    test_case.assertTrue(np.allclose(weight.grad.numpy(), weight_grad_np, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestEmbedding(flow.unittest.TestCase):
    def test_embedding(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_embedding_impl,
            _test_embedding_functional_impl,
            _test_embedding_renorm,
            _test_embedding_padding_idx,
            _test_embedding_scale_by_freq,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
