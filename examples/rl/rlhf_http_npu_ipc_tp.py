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

import os

import requests
import torch
import torch.distributed as dist
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferEngine,
)

# Use 127.0.0.1 (IPv4) rather than "localhost": when localhost resolves to the
# IPv6 ::1 first and the vLLM server only listens on IPv4 (0.0.0.0), requests
# hangs on the ::1 SYN until the connect timeout instead of falling back to
# IPv4 (the httpx-based OpenAI client may pick IPv4 and appear to work). This
# url is also what trainer_send_weights uses for its /update_weights POST.
BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Enable insecure serialization for IPC handle serialization over HTTP.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


def _bypass_proxy_for_localhost() -> None:
    """Ensure local hosts are excluded from any configured HTTP proxy.

    The vLLM server is local. If http_proxy/https_proxy is set in the
    environment, ``requests`` (used both here and inside ``trainer_send_weights``
    for the ``/update_weights`` POST) would route localhost through the proxy and
    time out on connect. Append the local hosts to no_proxy so every requests
    call -- including the engine's internal POST -- connects directly.
    """
    local_hosts = ["localhost", "127.0.0.1", "::1"]
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        entries = [e.strip() for e in current.split(",") if e.strip()]
        for host in local_hosts:
            if host not in entries:
                entries.append(host)
        os.environ[var] = ",".join(entries)


_bypass_proxy_for_localhost()

# Belt-and-suspenders: this session ignores proxy env entirely for the
# control-plane calls below regardless of how no_proxy is configured.
_SESSION = requests.Session()
_SESSION.trust_env = False

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


def init_weight_transfer_engine(base_url: str) -> None:
    """Initialize weight transfer via HTTP endpoint (no-op for NPU IPC)."""
    response = _SESSION.post(f"{base_url}/init_weight_transfer_engine", json={"init_info": {}}, timeout=60)
    response.raise_for_status()


def start_weight_update(base_url: str, is_checkpoint_format: bool = True) -> None:
    """Prepare layerwise reload on the vLLM server (call before update_weights)."""
    response = _SESSION.post(
        f"{base_url}/start_weight_update",
        json={"is_checkpoint_format": is_checkpoint_format},
        timeout=60,
    )
    response.raise_for_status()


def finish_weight_update(base_url: str) -> None:
    """Finalize layerwise reload on the vLLM server (call after update_weights)."""
    response = _SESSION.post(f"{base_url}/finish_weight_update", timeout=60)
    response.raise_for_status()


def pause_generation(base_url: str) -> None:
    response = _SESSION.post(f"{base_url}/pause", timeout=60)
    response.raise_for_status()


def resume_generation(base_url: str) -> None:
    response = _SESSION.post(f"{base_url}/resume", timeout=60)
    response.raise_for_status()


def main():
    # torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Each rank pins to its local
    # NPU chip; with ASCEND_RT_VISIBLE_DEVICES unset this maps logical index ==
    # physical chip == rank, matching the vLLM TP workers on chips 0..N-1.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"npu:{local_rank}"
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

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

    if rank == 0:
        print("-" * 50)
        print("Generating text BEFORE weight update (expect nonsense):")
        print("-" * 50)
        for prompt, text in zip(PROMPTS, generate_completions(client, MODEL_NAME, PROMPTS)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

        # Control-plane calls are driven by rank 0 only.
        init_weight_transfer_engine(BASE_URL)
        pause_generation(BASE_URL)
        start_weight_update(BASE_URL)

    # Barrier so start_weight_update completes before any rank POSTs weights.
    dist.barrier()

    # All ranks participate: handles are all-gathered/merged across ranks, then
    # rank 0 POSTs the merged payload to /update_weights.
    if rank == 0:
        print("Broadcasting weights via NPU IPC (HTTP, TP)...")
    trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode="http", url=BASE_URL)
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=trainer_args,
    )

    dist.barrier()

    if rank == 0:
        finish_weight_update(BASE_URL)
        resume_generation(BASE_URL)

        print("-" * 50)
        print("Generating text AFTER weight update:")
        print("-" * 50)
        for prompt, text in zip(PROMPTS, generate_completions(client, MODEL_NAME, PROMPTS)):
            print(f"Prompt: {prompt!r}\nGenerated text: {text!r}")
            print("-" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
