# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with NPU IPC-based weight syncing APIs.

Unlike rlhf_http_hccl.py which uses HCCL and can use separate NPUs, this script
uses Ascend NPU IPC which requires the training model and vLLM server to be on
the same physical NPU. Memory must be carefully managed to fit both models.

Prerequisites:
    Start a vLLM server with weight transfer enabled and reduced NPU memory
    utilization to leave room for the training model:

    $ VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        vllm serve Qwen/Qwen3-0.6b --enforce-eager \
        --weight-transfer-config '{"backend": "ipc"}' \
        --load-format dummy \
        --gpu-memory-utilization 0.5

    Then run this script:

    $ python rlhf_http_npu_ipc.py

The example performs the following steps:

* Load the training model on NPU 0 (same NPU as the vLLM server).
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint (no-op for NPU IPC).
* Pause generation and broadcast the real weights from the training model to
  the vLLM server using NPU IPC handles (via HTTP). The pause/resume is
  handled by ``trainer_send_weights`` — it calls ``update_weights`` internally.
* Generate text again to show normal output after the weight update.

For the tensor-parallel (TP > 1) workflow, see
``examples/rl/rlhf_http_npu_ipc_tp.py``.
"""

import base64
import os
import pickle

import httpx
import torch
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
# which uses it under the hood) connects instantly. So we route every call
# through httpx, including the ``/update_weights`` POST that the engine would
# otherwise make with ``requests`` (we pass a custom ``send_mode`` callable for
# that). 127.0.0.1 (IPv4) also avoids any localhost->::1 resolution stalls.
BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Enable insecure serialization for IPC handle serialization over HTTP
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# Shared httpx client, created in ``main`` BEFORE setting the NPU device and used
# as the OpenAI transport. A client constructed AFTER set_device (CANN init)
# cannot open new TCP connections to the local server (connect hangs); one
# constructed BEFORE can, including reconnects after an idle connection is
# dropped by the server's keep-alive timeout during the weight transfer.
_HTTP: httpx.Client | None = None


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
    """Start weight update via HTTP endpoint.

    Prepares the model for layerwise reload on the vLLM server side.
    Must be called before update_weights.
    """
    _HTTP.post(f"{BASE_URL}/start_weight_update", json={"is_checkpoint_format": is_checkpoint_format}).raise_for_status()


def finish_weight_update() -> None:
    """Finish weight update via HTTP endpoint.

    Finalizes layerwise reload on the vLLM server side.
    Must be called after all update_weights calls are complete.
    """
    _HTTP.post(f"{BASE_URL}/finish_weight_update").raise_for_status()


def pause_generation() -> None:
    """Pause generation via HTTP endpoint."""
    _HTTP.post(f"{BASE_URL}/pause").raise_for_status()


def resume_generation() -> None:
    """Resume generation via HTTP endpoint."""
    _HTTP.post(f"{BASE_URL}/resume").raise_for_status()


def is_multimodal_model(model_name: str) -> bool:
    """True if the HF config exposes a ``vision_config`` (multimodal model)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config")


def mapped_named_params(model: torch.nn.Module, is_multimodal: bool):
    """Yield (vllm_name, param). Multimodal models expose their language model
    under a ``language_model.`` prefix on the vLLM side."""
    for name, param in model.named_parameters():
        vllm_name = f"language_model.{name}" if is_multimodal else name
        yield vllm_name, param


def send_update_via_httpx(update_info) -> None:
    """Custom ``send_mode`` callable: POST ``/update_weights`` via httpx.

    Mirrors the engine's built-in ``send_mode="http"`` path but uses httpx
    instead of requests (see the module-level note on why requests breaks after
    setting the NPU device).
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
    # NPU IPC requires the training model to be on the same NPU as the vLLM server.
    # The server should be started on NPU 0 with reduced memory utilization.
    device = "npu:0"

    # Create the HTTP/OpenAI client BEFORE set_device (see the note at _HTTP), and
    # use it as the OpenAI transport so generation and control-plane share it.
    global _HTTP
    _HTTP = httpx.Client(trust_env=False, timeout=300.0, limits=httpx.Limits(keepalive_expiry=600.0))
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY", http_client=_HTTP)

    torch.accelerator.set_device_index(device)

    # Load the training model on the same NPU as the server.
    # Use bfloat16 to reduce memory footprint.
    print(f"Loading training model: {MODEL_NAME} on {device}")
    print(
        "Note: Ensure the vLLM server was started with --gpu-memory-utilization 0.5 "
        "or lower to leave room for the training model."
    )
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)
    train_model.eval()
    is_multimodal = is_multimodal_model(MODEL_NAME)

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Generate text before weight update. The output is expected to be nonsense
    # because the server is initialized with dummy weights.
    print("-" * 50)
    print("Generating text BEFORE weight update (expect nonsense):")
    print("-" * 50)
    outputs = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Initialize weight transfer on vLLM server (no-op for NPU IPC)
    print("Initializing weight transfer (NPU IPC backend)...")
    init_weight_transfer_engine()

    # Pause generation before weight sync
    pause_generation()

    # Start weight update (prepares layerwise reload on the vLLM server)
    start_weight_update()

    # Send weights via NPU IPC handles. We pass a custom ``send_mode`` callable
    # (httpx) instead of "http" so the /update_weights POST does not go through
    # requests, which breaks after torch.npu.set_device.
    print("Broadcasting weights via NPU IPC (HTTP)...")
    trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode=send_update_via_httpx, url=BASE_URL)
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=mapped_named_params(train_model, is_multimodal),
        trainer_args=trainer_args,
    )

    # Finish weight update (finalizes layerwise reload on the vLLM server)
    finish_weight_update()

    # Resume generation after weight sync
    resume_generation()

    # Generate text after weight update. The output is expected to be normal
    # because the real weights are now loaded.
    print("-" * 50)
    print("Generating text AFTER weight update:")
    print("-" * 50)
    outputs_updated = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs_updated):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Note: The training model and IPC handles remain in memory.
    # In a real RLHF training loop, you would update the training model
    # and create new IPC handles for each weight update.


if __name__ == "__main__":
    main()
