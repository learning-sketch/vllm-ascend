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
* The trainer holds a full (unsharded) copy of the model and broadcasts each
  tensor to all workers; every worker shards the full tensor locally on load.
  Do NOT pre-shard the trainer weights.

So this needs N + 1 NPUs total (N for inference + 1 for the trainer).

Prerequisites:
    Start a TP=N vLLM server with the HCCL weight-transfer backend. Pin it to
    chips ``0..N-1`` so the trainer can own chip ``N`` exclusively. Example for
    N=2 (needs 3 NPUs)::

        VLLM_SERVER_DEV_MODE=1 ASCEND_RT_VISIBLE_DEVICES=0,1 \
            vllm serve Qwen/Qwen3-0.6b --enforce-eager \
            --weight-transfer-config '{"backend": "nccl"}' \
            --load-format dummy \
            --tensor-parallel-size 2

    Then run this script (single process, no torchrun needed)::

        python rlhf_http_hccl_tp.py

The example generates text before and after the weight update to show the
server switching from dummy weights to the broadcast weights.

Note on HTTP transport:
    After setting the NPU device (CANN init) this process can no longer open new
    TCP connections to the local server from an HTTP client that was constructed
    AFTER set_device (connect hangs until timeout). So we construct the OpenAI
    client BEFORE set_device and reuse its httpx transport (``client._client``)
    for every control-plane / weight-update call. ``get_world_size`` runs before
    set_device too, so it uses a plain request.
"""

import threading

import httpx
import torch
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
)

BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Shared httpx client, set in ``main`` to the OpenAI client's OWN transport
# (``client._client``); see the module docstring on why this is required.
_HTTP: httpx.Client | None = None


def is_multimodal_model(model_name: str) -> bool:
    """True if the HF config exposes a ``vision_config`` (multimodal model)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config")


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    """Generate completions using the OpenAI-compatible API."""
    results = []
    for prompt in prompts:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=32, temperature=0)
        results.append(response.choices[0].text)
    return results


def get_world_size(base_url: str) -> int:
    """Get the inference world size from the vLLM server (TP * PP * DP).

    Called BEFORE the NPU device is set, so a plain request works here.
    """
    response = httpx.get(f"{base_url}/get_world_size", timeout=10.0)
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
    _HTTP.post(f"{BASE_URL}/update_weights", json={"update_info": update_info}, timeout=300.0).raise_for_status()


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
    # Query the inference world size (= TP * PP * DP) BEFORE setting the device.
    inference_world_size = get_world_size(BASE_URL)
    world_size = inference_world_size + 1  # +1 for the trainer (HCCL rank 0)

    # Create the OpenAI client BEFORE set_device and reuse its httpx transport for
    # every control-plane / weight-update call. This ordering is REQUIRED: a
    # client constructed AFTER the NPU device is set cannot open connections to
    # the local server afterwards (connect hangs until timeout), whereas one
    # constructed before connects fine even when the first request is sent later.
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")
    global _HTTP
    _HTTP = client._client

    # The trainer owns the chip just past the inference chips (0..N-1 are workers,
    # so the trainer uses chip N). This needs N + 1 NPUs total.
    device = f"npu:{inference_world_size}"
    torch.npu.set_device(inference_world_size)

    print(f"Loading training model: {MODEL_NAME} on {device} (inference_world_size={inference_world_size})")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)
    train_model.eval()
    # Multimodal models expose their language model under a ``language_model.``
    # prefix on the vLLM side; map the trainer param names accordingly.
    is_multimodal = is_multimodal_model(MODEL_NAME)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
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

    # Collect parameter metadata and size the packed buffer to fit the largest
    # tensor (+128 MiB headroom), keeping the 1 GiB default as a floor.
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0
    for name, p in train_model.named_parameters():
        names.append(f"language_model.{name}" if is_multimodal else name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))
        max_tensor_bytes = max(max_tensor_bytes, p.numel() * p.element_size())
    packed_buffer_size_bytes = max(max_tensor_bytes + 128 * 2**20, 2**30)

    # update_weights blocks on the server while it waits for HCCL broadcasts, so
    # run it in a thread while the trainer produces the data.
    update_thread = threading.Thread(
        target=update_weights,
        args=(names, dtype_names, shapes, True, packed_buffer_size_bytes),
    )
    update_thread.start()

    print("Broadcasting weights via HCCL (packed)...")
    trainer_args = HCCLTrainerSendWeightsArgs(
        group=model_update_group,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
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
