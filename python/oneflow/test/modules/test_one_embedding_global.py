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
from oneflow.test_utils.test_util import GenArgDict
import oneflow as flow
import numpy as np
import oneflow.nn as nn
import oneflow.unittest
import tempfile

from oneflow.test_utils.automated_test_util import *

path1 = "test1"
path2 = "test2"
path3 = "test3"


def get_param_list(parameters):
    param_list = []
    for x in parameters:
        param_list.append(x)
    return param_list


def get_optimizer_param(
    optimizer_test_case,
    norm_type,
    embedding_lookup1_params,
    embedding_lookup2_params,
    embedding_lookup3_params,
    dense1_params,
    dense2_params,
):
    embedding_lookup1_params = get_param_list(embedding_lookup1_params)
    embedding_lookup2_params = get_param_list(embedding_lookup2_params)
    embedding_lookup3_params = get_param_list(embedding_lookup3_params)
    dense1_params = get_param_list(dense1_params)
    dense2_params = get_param_list(dense2_params)
    if optimizer_test_case == 1:
        parameters1 = (
            embedding_lookup1_params
            + embedding_lookup2_params
            + embedding_lookup3_params
            + dense1_params
            + dense2_params
        )
        params = [
            {
                "params": parameters1,
                "clip_grad_max_norm": 0.5,
                "clip_grad_norm_type": norm_type,
            },
        ]
    elif optimizer_test_case == 2:
        # all grad in 1 group
        parameters1 = (
            embedding_lookup1_params + embedding_lookup2_params + dense1_params
        )
        parameters2 = embedding_lookup3_params + dense2_params
        params = [
            {
                "params": parameters1,
                "clip_grad_max_norm": 0.5,
                "clip_grad_norm_type": norm_type,
            },
            {"params": parameters2, "clip_grad_max_norm": 1, "clip_grad_norm_type": 2,},
        ]
    elif optimizer_test_case == 3:
        parameters1 = (
            embedding_lookup1_params
            + embedding_lookup2_params
            + dense1_params
            + dense2_params
        )
        parameters2 = embedding_lookup3_params
        params = [
            {
                "params": parameters1,
                "clip_grad_max_norm": 0.5,
                "clip_grad_norm_type": norm_type,
            },
            {"params": parameters2, "clip_grad_max_norm": 1, "clip_grad_norm_type": 2,},
        ]
    elif optimizer_test_case == 4:
        parameters1 = (
            embedding_lookup1_params
            + embedding_lookup2_params
            + dense1_params
            + dense2_params
        )
        parameters2 = embedding_lookup3_params
        params = [
            {"params": parameters1,},
            {"params": parameters2},
        ]
    return params


def _test_one_embedding(
    test_case,
    has_column_id,
    num_columns,
    use_fp16,
    loss_scale_policy,
    optimizer_test_case,
    norm_type,
):
    print(has_column_id, num_columns, use_fp16, optimizer_test_case, norm_type)
    placement = flow.placement(type="cuda", ranks=list(range(2)))
    batch_size = 4
    embedding_size = 2
    ids = np.random.randint(0, 1000, (batch_size, num_columns), dtype=np.int64)
    ids_tensor = flow.tensor(ids, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    if has_column_id:
        column_ids = (
            ids % num_columns
        )  # same id must have same column id, so in this case get column_ids from ids
        column_ids_tensor = flow.tensor(
            column_ids.astype(np.int32), requires_grad=False
        ).to_global(placement=placement, sbp=flow.sbp.split(0))
    else:
        column_ids_tensor = None

    class MatMul(flow.nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.w1 = flow.nn.Parameter(
                flow.randn(k, 1, placement=placement, sbp=flow.sbp.broadcast)
            )

        def forward(self, x):
            out = flow.matmul(x, self.w1)
            return out

    class OneEmbedding(nn.Module):
        def __init__(self, name, path):
            super().__init__()
            column_size_array = [np.random.randint(100, 1000)] * num_columns
            scales = np.sqrt(1 / np.array(column_size_array))
            initializer_list = []
            for i in range(scales.size):
                initializer_list.append(
                    {
                        "initializer": {
                            "type": "uniform",
                            "low": -scales[i],
                            "high": scales[i],
                        }
                    }
                )
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=16, persistent_path=path, size_factor=1,
            )
            self.embedding = flow.one_embedding.MultiTableEmbedding(
                name,
                embedding_size,
                flow.float,
                flow.int64,
                tables=initializer_list,
                store_options=store_options,
            )
            self.embedding = self.embedding.to_global(
                placement=placement, sbp=flow.sbp.broadcast
            )

        def forward(self, ids, column_ids):
            return self.embedding.forward(ids, column_ids)

    class TrainGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            if use_fp16:
                self.config.enable_amp(True)
                if loss_scale_policy == "static":
                    grad_scaler = flow.amp.StaticGradScaler(1024)
                else:
                    grad_scaler = flow.amp.GradScaler(
                        init_scale=1073741824,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=2000,
                    )
                self.set_grad_scaler(grad_scaler)
            self.dense1 = MatMul(embedding_size * num_columns, 1)
            self.dense2 = MatMul(embedding_size * num_columns, 1)
            self.embedding_lookup1 = OneEmbedding("emb1", path1)
            self.embedding_lookup2 = OneEmbedding("emb2", path2)
            self.embedding_lookup3 = OneEmbedding("emb3", path3)

            params = get_optimizer_param(
                optimizer_test_case,
                norm_type,
                self.embedding_lookup1.parameters(),
                self.embedding_lookup2.parameters(),
                self.embedding_lookup3.parameters(),
                self.dense1.parameters(),
                self.dense2.parameters(),
            )
            self.add_optimizer(flow.optim.SGD(params, lr=0.1, momentum=0.0))

        def build(self, ids, column_ids):
            embedding1 = self.embedding_lookup1.forward(ids, column_ids)
            embedding2 = self.embedding_lookup2.forward(ids, column_ids)
            embedding3 = self.embedding_lookup3.forward(ids, column_ids)
            embedding = embedding1 + embedding2
            embedding = embedding + embedding3
            loss = embedding.reshape(embedding.shape[0], -1)
            loss1 = self.dense1(loss)
            loss2 = self.dense2(loss)
            loss = loss1 + loss2
            loss = loss.mean()
            loss.backward()
            return loss

    graph = TrainGraph()
    loss = graph(ids_tensor, column_ids_tensor)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class OneEmbeddingTestCase(flow.unittest.TestCase):
    def test_one_embedding2(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True]
        arg_dict["num_columns"] = [26]
        arg_dict["use_fp16"] = [False]  # [True, False]
        arg_dict["loss_scale_policy"] = ["dynamic"]  # ["static", "dynamic"]
        arg_dict["optimizer_test_case"] = [4]  # [1,2,3,4,5]
        arg_dict["norm_type"] = [2]  # , np.inf, -np.inf]
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
