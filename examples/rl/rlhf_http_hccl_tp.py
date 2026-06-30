# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF weight syncing with HCCL under tensor parallelism (TP > 1).

This is the tensor-parallel counterpart of ``rlhf_http_hccl.py``. Unlike NPU IPC
(which requires the trainer to be co-located on the same physical chip as each
vLLM worker, hence one trainer rank per chip), HCCL broadcasts the weights over a
process group, so a SINGLE trainer process can update an arbitrary
``--tensor-parallel-size N`` server:

* The vLLM server runs N workers on chips ``0..N-1`` (HCCL ranks ``1..N``).
* The trainer is a single process on its own chip ``N`` (HCCL rank ``0``).
* The trainer broadcasts each full (unsharded) tensor to all workers; every
  worker shards the full tensor locally on load. Do NOT pre-shard the weights.

So this needs N + 1 NPUs total (N for inference + 1 for the trainer).

Memory: the trainer model stays on CPU and each parameter is streamed to the NPU
just before it is broadcast (and freed after), so the full model is never
resident on the trainer's NPU -- only the packed broadcast buffer plus the
current tensor. This lets a single trainer NPU drive a model far larger than one
chip (the inference side is still sharded across the N workers).

Fused MoE: recent HF MoE checkpoints (e.g. Qwen3-MoE) store all experts of a
layer in fused tensors (``experts.gate_up_proj`` / ``experts.down_proj``), which
vLLM's FusedMoE loader cannot match. They are expanded on the fly into the
per-expert names vLLM expects (``experts.{e}.gate_proj/up_proj/down_proj.weight``).

Prerequisites:
    Start a TP=N vLLM server with the HCCL weight-transfer backend, pinned to
    chips ``0..N-1`` so the trainer can own chip ``N``. Start it with
    ``--enforce-eager`` (aclgraph + layerwise reload can otherwise fail/hang) and,
    for MoE, the worker-side patch that preserves the fused-expert
    ``weight_loader`` across updates (see
    ``examples/rl/npu_moe_weight_loader_patch.py``). Example for N=4::

        PYTHONPATH=examples/rl:$PYTHONPATH \
        VLLM_SERVER_DEV_MODE=1 VLLM_ASCEND_ENABLE_NZ=0 \
            ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
            vllm serve /path/to/Qwen3.5-35B-A3B --enforce-eager \
            --weight-transfer-config '{"backend": "nccl"}' \
            --load-format dummy \
            --tensor-parallel-size 4 --enable-expert-parallel \
            --gpu-memory-utilization 0.5 \
            --worker-extension-cls npu_moe_weight_loader_patch.MoEWeightLoaderWorkerExtension

    Then run this script (single process, no torchrun needed). It uses NPU chip
    ``N`` (= inference_world_size), so make that chip visible to the trainer::

        MODEL_NAME=/path/to/Qwen3.5-35B-A3B python rlhf_http_hccl_tp.py

Note on HTTP transport:
    After setting the NPU device (CANN init) this process can no longer open new
    TCP connections to the local server from a client constructed AFTER
    set_device. So we construct the httpx/OpenAI client BEFORE set_device and
    reuse it for every control-plane / weight-update call.
"""

import os
import threading
from collections.abc import Iterator

import httpx
import torch
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
# Override with e.g. MODEL_NAME=/path/to/model ; must match the served model.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
# USE_CHAT=1 -> /v1/chat/completions (applies the chat template); else /v1/completions.
USE_CHAT = os.environ.get("USE_CHAT", "0") == "1"

PACKED_BUFFER_HEADROOM_BYTES = 128 * 2**20  # 128 MiB on top of the largest tensor

# Shared httpx client, created in ``main`` BEFORE set_device and used as the
# OpenAI transport; see the module docstring on why this ordering is required.
_HTTP: httpx.Client | None = None


def is_multimodal_model(model_name: str) -> bool:
    """True if the HF config exposes a ``vision_config`` (multimodal model)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config")


def iter_vllm_named_params(
    model: torch.nn.Module, is_multimodal: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (vllm_name, cpu_tensor) for every weight, expanding fused MoE experts.

    Tensors are CPU views (no NPU copy here) so this is cheap to iterate twice:
    once to collect the names/shapes metadata and once to broadcast. Both passes
    MUST yield the exact same sequence so the packed producer (trainer) and
    consumer (server) agree on chunk boundaries -- they do, since this is
    deterministic for a given model.

    Multimodal models expose their language model under a ``language_model.``
    prefix on the vLLM side. Fused MoE experts are split into per-expert names.
    """
    prefix = "language_model." if is_multimodal else ""
    for name, param in model.named_parameters():
        p = param.detach()
        if name.endswith("mlp.experts.gate_up_proj"):
            base = prefix + name[: -len("gate_up_proj")]  # ...mlp.experts.
            half = p.shape[1] // 2  # rows [:half] = gate, [half:] = up
            for e in range(p.shape[0]):
                yield f"{base}{e}.gate_proj.weight", p[e, :half, :]
                yield f"{base}{e}.up_proj.weight", p[e, half:, :]
        elif name.endswith("mlp.experts.down_proj"):
            base = prefix + name[: -len("down_proj")]
            for e in range(p.shape[0]):
                yield f"{base}{e}.down_proj.weight", p[e]
        else:
            yield prefix + name, p


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    """Generate via /v1/chat/completions when USE_CHAT, else /v1/completions."""
    results = []
    for prompt in prompts:
        if USE_CHAT:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0,
            )
            results.append(response.choices[0].message.content)
        else:
            response = client.completions.create(model=model, prompt=prompt, max_tokens=32, temperature=0)
            results.append(response.choices[0].text)
    return results


def get_world_size() -> int:
    """Get the inference world size from the vLLM server (TP * PP * DP)."""
    response = _HTTP.get(f"{BASE_URL}/get_world_size", timeout=10.0)
    response.raise_for_status()
    return response.json()["world_size"]


def init_weight_transfer_engine(
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
) -> None:
    """Initialize weight transfer on the server (blocks until HCCL connects)."""
    payload = {
        "init_info": dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
        )
    }
    _HTTP.post(f"{BASE_URL}/init_weight_transfer_engine", json=payload, timeout=120.0).raise_for_status()


def update_weights(
    names: list[str],
    dtype_names: list[str],
    shapes: list[list[int]],
    packed: bool = False,
    packed_buffer_size_bytes: int | None = None,
) -> None:
    """Trigger the server-side receive (blocks while it waits for HCCL broadcasts)."""
    update_info = dict(names=names, dtype_names=dtype_names, shapes=shapes, packed=packed)
    if packed and packed_buffer_size_bytes is not None:
        update_info["packed_buffer_size_bytes"] = packed_buffer_size_bytes
    _HTTP.post(f"{BASE_URL}/update_weights", json={"update_info": update_info}, timeout=600.0).raise_for_status()


def start_weight_update(is_checkpoint_format: bool = True) -> None:
    """Prepare layerwise reload on the server (call before update_weights)."""
    _HTTP.post(f"{BASE_URL}/start_weight_update", json={"is_checkpoint_format": is_checkpoint_format}).raise_for_status()


def finish_weight_update() -> None:
    """Finalize layerwise reload on the server (call after update_weights)."""
    _HTTP.post(f"{BASE_URL}/finish_weight_update").raise_for_status()


def pause_generation() -> None:
    _HTTP.post(f"{BASE_URL}/pause").raise_for_status()


def resume_generation() -> None:
    _HTTP.post(f"{BASE_URL}/resume").raise_for_status()


def main():
    # Create the OpenAI/httpx client BEFORE set_device (see the module docstring),
    # and reuse it for every control-plane / weight-update call.
    global _HTTP
    _HTTP = httpx.Client(trust_env=False, timeout=600.0, limits=httpx.Limits(keepalive_expiry=600.0))
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY", http_client=_HTTP)

    inference_world_size = get_world_size()
    world_size = inference_world_size + 1  # +1 for the trainer (HCCL rank 0)

    # The trainer owns the chip just past the inference chips (0..N-1 are workers).
    device = f"npu:{inference_world_size}"
    torch.npu.set_device(inference_world_size)

    # Keep the trainer model on CPU; stream each parameter to the NPU only when it
    # is broadcast (see iter_vllm_named_params / the broadcast iterator below).
    print(f"Loading training model on CPU (streamed to {device}): {MODEL_NAME}")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.eval()
    is_multimodal = is_multimodal_model(MODEL_NAME)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    print("-" * 50)
    print("Generating text BEFORE weight update (expect nonsense):")
    print("-" * 50)
    for prompt, text in zip(prompts, generate_completions(client, MODEL_NAME, prompts)):
        print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
        print("-" * 50)

    # Set up the HCCL group: trainer is rank 0, workers start at rank_offset=1.
    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1
    print(f"Initializing weight transfer: master={master_address}:{master_port}, world_size={world_size}")

    # The server-side init blocks until the trainer connects, so run it in a
    # thread while the trainer builds its side of the HCCL group.
    init_thread = threading.Thread(
        target=init_weight_transfer_engine,
        args=(master_address, master_port, rank_offset, world_size),
    )
    init_thread.start()

    model_update_group = HCCLWeightTransferEngine.trainer_init(
        dict(master_address=master_address, master_port=master_port, world_size=world_size),
    )
    init_thread.join()

    # Pause generation and start the weight-update lifecycle.
    pause_generation()
    start_weight_update()

    # Collect parameter metadata (post MoE expansion) and size the packed buffer
    # to fit the largest single tensor (+headroom), keeping a 1 GiB floor.
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0
    for name, tensor in iter_vllm_named_params(train_model, is_multimodal):
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        max_tensor_bytes = max(max_tensor_bytes, tensor.numel() * tensor.element_size())
    packed_buffer_size_bytes = max(max_tensor_bytes + PACKED_BUFFER_HEADROOM_BYTES, 2**30)

    # update_weights blocks on the server while it waits for HCCL broadcasts, so
    # run it in a thread while the trainer produces the data.
    update_thread = threading.Thread(
        target=update_weights,
        args=(names, dtype_names, shapes, True, packed_buffer_size_bytes),
    )
    update_thread.start()

    # Broadcast in the SAME order as the metadata above, streaming each tensor to
    # the NPU just-in-time (the packed producer frees each chunk after broadcast).
    def broadcast_iter() -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in iter_vllm_named_params(train_model, is_multimodal):
            yield name, tensor.to(device)

    print(f"Broadcasting weights via HCCL (packed, buffer={packed_buffer_size_bytes / 2**20:.0f} MiB)...")
    trainer_args = HCCLTrainerSendWeightsArgs(
        group=model_update_group,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=broadcast_iter(),
        trainer_args=trainer_args,
    )
    update_thread.join()

    # Finalize the lifecycle and resume generation.
    finish_weight_update()
    resume_generation()

    print("-" * 50)
    print("Generating text AFTER weight update:")
    print("-" * 50)
    for prompt, text in zip(prompts, generate_completions(client, MODEL_NAME, prompts)):
        print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
