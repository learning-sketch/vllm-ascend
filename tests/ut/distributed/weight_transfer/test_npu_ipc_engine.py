#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for the NPU IPC weight transfer engine.

These cover two bugs that broke ``examples/rl/rlhf_http_npu_ipc.py``:

1. ``NPUIPCWeightTransferEngine.__init__`` did not accept the ``model``
   argument that ``WeightTransferEngineFactory.create_engine`` passes,
   raising ``TypeError: __init__() takes 3 positional arguments but 4
   were given`` at engine construction.
2. The trainer side stored only the ``reduce_tensor`` *args* (dropping the
   rebuild *func*), while ``receive_weights`` unpacked the stored value as
   ``func, args``, raising ``ValueError: too many values to unpack
   (expected 2)`` during ``update_weights``.
"""

import inspect
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCWeightTransferEngine,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


def test_init_accepts_model_argument():
    """Bug 1: __init__ must accept the optional ``model`` argument."""
    params = inspect.signature(NPUIPCWeightTransferEngine.__init__).parameters
    assert "model" in params


def test_init_passes_model_to_super():
    """Bug 1: the ``model`` argument must be forwarded to the base engine."""
    captured = {}

    def fake_init(self, config, parallel_config, model=None):
        captured["args"] = (config, parallel_config, model)

    with patch.object(npu_ipc_engine.WeightTransferEngine, "__init__", fake_init):
        NPUIPCWeightTransferEngine("config", "parallel_config", "model")

    assert captured["args"] == ("config", "parallel_config", "model")


def test_unpacked_send_keeps_full_reduce_tensor_handle():
    """Bug 2: the stored IPC handle must be the full ``(func, args)`` tuple.

    Drives the unpacked trainer send and verifies the handle dict carries the
    complete ``reduce_tensor`` result, then feeds the produced update info back
    into ``receive_weights`` to confirm the round-trip unpacking works.
    """
    npu_uuid = "node-0"
    device_index = 0

    rebuilt_weight = torch.tensor([1.0, 2.0, 3.0])

    def rebuild_func(*args):
        # Index 6 is the device index, which receive_weights overwrites.
        assert args[6] == device_index
        return rebuilt_weight

    # ``reduce_tensor`` returns (rebuild_func, rebuild_args). The args tuple is
    # longer than 2, which is precisely why dropping the func and unpacking the
    # args as ``func, args`` used to raise "too many values to unpack".
    rebuild_args = (None, None, None, None, None, None, 999, None)
    fake_reduce = MagicMock(return_value=(rebuild_func, rebuild_args))

    captured = {}

    def send_mode(update_info):
        captured["update_info"] = update_info

    trainer_args = MagicMock()
    trainer_args.send_mode = send_mode
    trainer_args.packed = False

    iterator = iter([("model.weight", torch.zeros(3))])

    with patch(f"{_MODULE}.reduce_tensor", fake_reduce):
        NPUIPCWeightTransferEngine._send_unpacked(iterator, trainer_args, npu_uuid)

    update_info = captured["update_info"]
    assert isinstance(update_info.ipc_handles, list)
    stored = update_info.ipc_handles[0][npu_uuid]
    # Must be the full (func, args) tuple, not just the args.
    assert stored == (rebuild_func, rebuild_args)

    # Round-trip: receive_weights should unpack and rebuild without error.
    engine = object.__new__(NPUIPCWeightTransferEngine)
    received = {}

    def load_weights(weights):
        received["weights"] = weights

    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid),
        patch("torch.accelerator.current_device_index", return_value=device_index),
    ):
        engine.receive_weights(update_info, load_weights)

    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
