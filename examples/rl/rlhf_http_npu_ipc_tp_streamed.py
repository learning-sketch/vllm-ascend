# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF weight syncing with NPU IPC under TP, for models larger than a single chip.

This builds on ``rlhf_http_npu_ipc_tp.py``. The plain TP example keeps a FULL
(replicated) copy of the model resident on every trainer chip, so the peak
per-chip memory is ``full_model + worker_shard + KV``. For a model that does not
fit alongside the inference worker on one chip, that full replicated copy is the
bottleneck.

This example removes the full resident copy by *streaming* the weights:

* The trainer model stays on CPU (it is never moved to the NPU in one piece).
* A lazy iterator yields ONE parameter at a time, moving just that parameter to
  the NPU just-in-time and freeing it right after.
* ``packed=True`` makes the IPC transfer reuse a single fixed-size NPU buffer
  (``packed_buffer_size_bytes``), copying each chunk in and POSTing it
  synchronously before the buffer is overwritten.

So the trainer's NPU footprint is bounded by roughly ``one parameter + one
packed buffer`` instead of the whole model. This mirrors how vLLM's official RL
framework (``vllm-project/vime``) streams Megatron weights to colocated engines:
gather/produce full weights in bounded buckets, send, free, repeat.

Where to plug in a real sharded trainer:
    Here the per-parameter "full tensor" is produced by moving a CPU parameter to
    the NPU (``_streamed_full_param_iter``). In a real TP/FSDP/Megatron trainer
    whose shards already live on the NPU, replace that line with an all-gather of
    the shards across your trainer's parallel group to reconstruct the full
    tensor on the fly -- the rest of the pipeline is identical.

Prerequisites (TP=2 shown; needs 2 NPUs):

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        ASCEND_RT_VISIBLE_DEVICES=0,1 \
        vllm serve Qwen/Qwen3-0.6b --enforce-eager \
        --weight-transfer-config '{"backend": "ipc"}' \
        --load-format dummy \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.7

    torchrun --nproc-per-node=2 rlhf_http_npu_ipc_tp_streamed.py
"""

import base64
import os
import pickle
from collections.abc import Iterator

import httpx
import torch
import torch.distributed as dist
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferEngine,
)

BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# Headroom added on top of the largest single (full) parameter when sizing the
# packed IPC buffer. The buffer must be at least as large as the biggest tensor.
PACKED_BUFFER_HEADROOM_BYTES = 128 * 2**20  # 128 MiB

# Shared httpx client, set in ``main`` to the OpenAI client's OWN transport; see
# rlhf_http_npu_ipc_tp.py for why a hand-built client cannot connect post-set_device.
_HTTP: httpx.Client | None = None


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    results = []
    for prompt in prompts:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=32, temperature=0)
        results.append(response.choices[0].text)
    return results


def init_weight_transfer_engine() -> None:
    _HTTP.post(f"{BASE_URL}/init_weight_transfer_engine", json={"init_info": {}}).raise_for_status()


def start_weight_update(is_checkpoint_format: bool = True) -> None:
    _HTTP.post(f"{BASE_URL}/start_weight_update", json={"is_checkpoint_format": is_checkpoint_format}).raise_for_status()


def finish_weight_update() -> None:
    _HTTP.post(f"{BASE_URL}/finish_weight_update").raise_for_status()


def pause_generation() -> None:
    _HTTP.post(f"{BASE_URL}/pause").raise_for_status()


def resume_generation() -> None:
    _HTTP.post(f"{BASE_URL}/resume").raise_for_status()


def send_update_via_httpx(update_info) -> None:
    """Custom ``send_mode`` callable: POST ``/update_weights`` via httpx.

    Handles both packed and non-packed update_info (``ipc_handles`` is a dict in
    packed mode, a list otherwise); ``pickle`` serializes either form.
    """
    fields = {
        "names": update_info.names,
        "dtype_names": update_info.dtype_names,
        "shapes": update_info.shapes,
        "packed": update_info.packed,
    }
    if update_info.tensor_sizes is not None:
        fields["tensor_sizes"] = update_info.tensor_sizes
    fields["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(update_info.ipc_handles)).decode("utf-8")
    _HTTP.post(f"{BASE_URL}/update_weights", json={"update_info": fields}).raise_for_status()


def is_multimodal_model(model_name: str) -> bool:
    """True if the HF config exposes a ``vision_config`` (multimodal model)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config")


def _streamed_full_param_iter(
    model: torch.nn.Module, device: str, is_multimodal: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield full parameters one at a time, materialized on the NPU just-in-time.

    The model stays on CPU, so the FULL model is never resident on the NPU --
    only the current parameter (plus the packed transfer buffer) is on the NPU.
    The packed producer copies each tensor into its reusable buffer, after which
    this generator frees the NPU copy before producing the next one.

    Multimodal models expose their language model under a ``language_model.``
    prefix on the vLLM side, so the name is mapped accordingly.

    For a real TP/FSDP/Megatron trainer whose shards live on the NPU, replace the
    ``.to(device)`` below with an all-gather of the shards across your trainer's
    parallel group to reconstruct the full tensor here.
    """
    for name, param in model.named_parameters():
        vllm_name = f"language_model.{name}" if is_multimodal else name
        full = param.detach().to(device)
        yield vllm_name, full
        del full


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"npu:{local_rank}"
    torch.npu.set_device(local_rank)

    # gloo group (CPU/TCP) so the packed path can all-gather/merge IPC handles
    # across ranks; the weight data itself moves via IPC shared memory.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Keep the trainer model on CPU -- do NOT move it to the NPU. Only one
    # parameter at a time is streamed to the NPU during the transfer.
    if rank == 0:
        print(f"Loading training model on CPU (streamed to NPU per-param): {MODEL_NAME}")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.eval()
    is_multimodal = is_multimodal_model(MODEL_NAME)

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")
    global _HTTP
    _HTTP = client._client

    # Size the packed buffer to fit the largest single (full) parameter, with
    # headroom. The buffer is the dominant bounded NPU cost of the transfer.
    max_param_bytes = max(p.numel() * p.element_size() for _, p in train_model.named_parameters())
    packed_buffer_size_bytes = max_param_bytes + PACKED_BUFFER_HEADROOM_BYTES

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    if rank == 0:
        print("-" * 50)
        print("Generating text BEFORE weight update (expect nonsense):")
        print("-" * 50)
        for prompt, text in zip(prompts, generate_completions(client, MODEL_NAME, prompts)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

        init_weight_transfer_engine()
        pause_generation()
        start_weight_update()

    dist.barrier()

    if rank == 0:
        print(f"Streaming weights via NPU IPC (packed, buffer={packed_buffer_size_bytes / 2**20:.0f} MiB)...")
    trainer_args = NPUIPCTrainerSendWeightsArgs(
        send_mode=send_update_via_httpx,
        url=BASE_URL,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=_streamed_full_param_iter(train_model, device, is_multimodal),
        trainer_args=trainer_args,
    )

    dist.barrier()

    if rank == 0:
        finish_weight_update()
        resume_generation()

        print("-" * 50)
        print("Generating text AFTER weight update:")
        print("-" * 50)
        for prompt, text in zip(prompts, generate_completions(client, MODEL_NAME, prompts)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
