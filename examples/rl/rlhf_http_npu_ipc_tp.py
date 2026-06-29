# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF weight syncing with NPU IPC under tensor parallelism (TP > 1).

This is the tensor-parallel counterpart of ``rlhf_http_npu_ipc.py``. NPU IPC
requires the trainer rank and its vLLM inference worker to be co-located on the
*same physical NPU chip*. With ``--tensor-parallel-size N`` the vLLM server
spawns ``N`` workers on physical chips ``0..N-1``, so the trainer must also run
as ``N`` ranks, one pinned to each of those same chips.

Key points:

* Each trainer rank holds a *full (replicated)* copy of the model. Do NOT
  pre-shard the trainer weights: vLLM shards each full tensor locally when it
  loads it, so every worker needs the full tensor from its co-located rank.
* ``trainer_send_weights`` all-gathers and merges the per-chip IPC handles
  across ranks so every vLLM worker can find the handle for its own chip.
* Only rank 0 drives the HTTP control plane (pause / start_weight_update /
  update_weights / finish_weight_update / resume); the POST to
  ``/update_weights`` happens inside ``trainer_send_weights`` on rank 0.

Prerequisites:
    Start a TP=N vLLM server with weight transfer enabled and reduced NPU
    memory utilization to leave room for the (replicated) training model.
    Pin it to physical chips 0..N-1 so its per-worker IPC UUIDs match the
    trainer ranks. Example for N=2::

        VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
            ASCEND_RT_VISIBLE_DEVICES=0,1 \
            vllm serve Qwen/Qwen3-0.6b --enforce-eager \
            --weight-transfer-config '{"backend": "ipc"}' \
            --load-format dummy \
            --tensor-parallel-size 2 \
            --gpu-memory-utilization 0.5

    Then launch this script with one trainer process per chip via torchrun.
    Leave ASCEND_RT_VISIBLE_DEVICES unset for the trainer so logical index ==
    physical chip == rank, matching the vLLM workers::

        torchrun --nproc-per-node=2 rlhf_http_npu_ipc_tp.py

The script generates text before and after the weight update (rank 0 only) to
show the server switching from dummy weights to the broadcast weights.

Note on the trainer process group backend:
    The trainer group uses the ``gloo`` (CPU/TCP) backend, not ``hccl``. It only
    needs to all-gather the CPU-side IPC handles and barrier; the weight data
    itself moves via IPC shared memory. Because the vLLM server already holds an
    HCCL communicator on these same physical chips, starting a second ``hccl``
    group here would collide on the NPU socket port and fail with
    ``HCCL function error ... code is 7`` / ``EI0020 ... port already bound``.
    (If you must use ``hccl`` for some reason, give the trainer a non-overlapping
    ``HCCL_NPU_SOCKET_PORT_RANGE`` instead.)
"""

import base64
import os
import pickle

import httpx
import torch
import torch.distributed as dist
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferEngine,
)

# IMPORTANT: use httpx, not requests, for all trainer -> server HTTP.
#
# After ``torch.npu.set_device`` (CANN/torch_npu init), ``requests``/urllib3 can
# no longer open NEW TCP connections to the local server -- ``connect`` hangs
# until timeout -- even though a raw ``socket.create_connection`` (and httpx,
# which uses it under the hood) connects instantly. urllib3 ships its own
# ``create_connection`` implementation that breaks in this environment, so we
# route every call through httpx, including the ``/update_weights`` POST that the
# engine would otherwise make with ``requests`` (we pass a custom ``send_mode``
# callable for that). Use 127.0.0.1 (IPv4) to also avoid any localhost->::1
# resolution stalls when the server only listens on IPv4.
BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Enable insecure serialization for IPC handle serialization over HTTP.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# Shared httpx client, created in ``main`` BEFORE torch.npu.set_device and used
# as the OpenAI transport. A client constructed AFTER set_device (CANN init)
# cannot open new TCP connections to the local server (connect hangs); one
# constructed BEFORE can, including reconnects after an idle connection is
# dropped by the server's keep-alive timeout during the weight transfer.
_HTTP: httpx.Client | None = None

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    """Generate completions using the OpenAI-compatible API."""
    results = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=32,
            temperature=0,
        )
        results.append(response.choices[0].text)
    return results


def init_weight_transfer_engine() -> None:
    """Initialize weight transfer via HTTP endpoint (no-op for NPU IPC)."""
    _HTTP.post(f"{BASE_URL}/init_weight_transfer_engine", json={"init_info": {}}).raise_for_status()


def start_weight_update(is_checkpoint_format: bool = True) -> None:
    """Prepare layerwise reload on the vLLM server (call before update_weights)."""
    _HTTP.post(f"{BASE_URL}/start_weight_update", json={"is_checkpoint_format": is_checkpoint_format}).raise_for_status()


def finish_weight_update() -> None:
    """Finalize layerwise reload on the vLLM server (call after update_weights)."""
    _HTTP.post(f"{BASE_URL}/finish_weight_update").raise_for_status()


def pause_generation() -> None:
    _HTTP.post(f"{BASE_URL}/pause").raise_for_status()


def resume_generation() -> None:
    _HTTP.post(f"{BASE_URL}/resume").raise_for_status()


def is_multimodal_model(model_name: str) -> bool:
    """True if the HF config exposes a ``vision_config`` (multimodal model)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config")


def mapped_named_params(model: torch.nn.Module, is_multimodal: bool):
    """Yield (vllm_name, param). For multimodal models the HF language-model
    parameters live under the ``language_model.`` prefix on the vLLM side, so we
    add it; for text-only models the names are passed through unchanged."""
    for name, param in model.named_parameters():
        vllm_name = f"language_model.{name}" if is_multimodal else name
        yield vllm_name, param


def send_update_via_httpx(update_info) -> None:
    """Custom ``send_mode`` callable: POST ``/update_weights`` via httpx.

    Mirrors the engine's built-in ``send_mode="http"`` path, but uses httpx
    instead of requests (see the module docstring on why requests breaks after
    ``torch.npu.set_device``). The IPC handles are pickled + base64-encoded under
    ``ipc_handles_pickled``; the server deserializes them with
    ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``.
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


def main():
    # torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Each rank pins to its local
    # NPU chip; with ASCEND_RT_VISIBLE_DEVICES unset this maps logical index ==
    # physical chip == rank, matching the vLLM TP workers on chips 0..N-1.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"npu:{local_rank}"

    # Create the HTTP/OpenAI client BEFORE set_device (see the note at _HTTP), and
    # use it as the OpenAI transport so generation and control-plane share it.
    global _HTTP
    _HTTP = httpx.Client(trust_env=False, timeout=300.0, limits=httpx.Limits(keepalive_expiry=600.0))
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY", http_client=_HTTP)

    torch.npu.set_device(local_rank)

    # The trainer ranks form their own process group (independent of vLLM's
    # internal TP group) so trainer_send_weights can all-gather IPC handles.
    #
    # Use the gloo (CPU/TCP) backend, NOT hccl: this group only all-gathers the
    # (CPU-side) IPC handle metadata and barriers — the actual weight data moves
    # via IPC shared memory, not through this group. The vLLM server already
    # holds an HCCL communicator on these same physical chips, so creating a
    # second hccl group here collides on the NPU socket port and fails with
    # "HCCL function error ... code is 7" / EI0020 (port already bound).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Each rank loads a full (replicated) copy of the model. bfloat16 keeps the
    # footprint small enough to share the chip with the vLLM worker.
    if rank == 0:
        print(f"Loading replicated training model: {MODEL_NAME} (world_size={world_size})")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)
    train_model.eval()

    # Multimodal models expose their language model under a ``language_model.``
    # prefix on the vLLM side; map the trainer param names accordingly.
    is_multimodal = is_multimodal_model(MODEL_NAME)

    if rank == 0:
        print("-" * 50)
        print("Generating text BEFORE weight update (expect nonsense):")
        print("-" * 50)
        for prompt, text in zip(PROMPTS, generate_completions(client, MODEL_NAME, PROMPTS)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

        # Control-plane calls are driven by rank 0 only.
        init_weight_transfer_engine()
        pause_generation()
        start_weight_update()

    # Barrier so start_weight_update completes before any rank POSTs weights.
    dist.barrier()

    # All ranks participate: handles are all-gathered/merged across ranks, then
    # rank 0 POSTs the merged payload to /update_weights. We pass a custom
    # ``send_mode`` callable (httpx) instead of "http" so the POST does not go
    # through requests, which breaks after torch.npu.set_device.
    if rank == 0:
        print("Broadcasting weights via NPU IPC (HTTP, TP)...")
    trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode=send_update_via_httpx, url=BASE_URL)
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=mapped_named_params(train_model, is_multimodal),
        trainer_args=trainer_args,
    )

    dist.barrier()

    if rank == 0:
        finish_weight_update()
        resume_generation()

        print("-" * 50)
        print("Generating text AFTER weight update:")
        print("-" * 50)
        for prompt, text in zip(PROMPTS, generate_completions(client, MODEL_NAME, PROMPTS)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
