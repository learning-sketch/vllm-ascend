#
# Standalone script to verify the accuracy of sleep mode level 2 on Ascend NPU.
#
# It does NOT depend on the vLLM source repo, only on installed packages:
#   - vllm
#   - vllm-ascend
#   - torch / torch_npu
#   - safetensors
#
# Idea:
#   1. Build an LLM with enable_sleep_mode=True (external_launcher backend so the
#      model object lives in this process and we can reload its weights).
#   2. Run greedy generation (temperature=0) to get a deterministic baseline.
#   3. llm.sleep(level=2): discard BOTH weights and kv cache from device memory
#      (level 2 does NOT offload weights to CPU, unlike level 1).
#   4. Wake up: restore the weight buffers, reload weights from the safetensors
#      files on disk, then restore the kv cache.
#   5. Run the same greedy generation again and compare the outputs token-by-token.
#      If sleep mode 2 is correct, the outputs must be identical.
#
# Usage (single card, dense model such as Qwen3-0.6B):
#   python verify_sleep_mode2_accuracy.py --model /path/to/Qwen3-0.6B
#
# If --model is a modelscope/HF id instead of a local dir, the script will try to
# download it first (requires network / modelscope).

import argparse
import os

import torch

# Keep behaviour close to the tested vllm-ascend example.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# NZ weight format is not compatible with the level-2 weight reload path.
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

from safetensors.torch import load_file  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def patch_vllm_moe_model_weight_loader(model):
    """Re-attach the fused MoE weight loader after a fresh model is created.

    Only needed for MoE models; harmless to call for dense models because the
    inner model will simply not contain w13_weight / w2_weight parameters.
    """
    inner = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner is None or not hasattr(inner, "layers"):
        return
    for layer in inner.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "experts"):
            continue
        for name, param in dict(mlp.named_parameters()).items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader


def load_and_merge_safetensors(directory):
    """Merge all *.safetensors shards in a directory into a single state dict."""
    if not os.path.isdir(directory):
        raise ValueError(f"Not a local directory with weights: {directory}")
    merged = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            print(f"[reload] loading shard: {file_path}")
            merged.update(load_file(file_path))
    if not merged:
        raise ValueError(f"No *.safetensors files found in {directory}")
    return merged


def resolve_model_dir(model: str) -> str:
    """Return a local directory that contains the safetensors weights."""
    if os.path.isdir(model):
        return model
    # Try to download from modelscope (the example uses VLLM_USE_MODELSCOPE).
    print(f"[model] '{model}' is not a local dir, attempting download...")
    try:
        from modelscope import snapshot_download  # type: ignore

        return snapshot_download(model)
    except Exception:
        from huggingface_hub import snapshot_download  # type: ignore

        return snapshot_download(model)


def get_runner_model(llm: LLM):
    """Reach the underlying nn.Module via the external_launcher driver worker."""
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model


def parse_args():
    parser = argparse.ArgumentParser(description="Verify sleep mode level 2 accuracy")
    parser.add_argument("--model", required=True, help="Local dir or model id (dense model, e.g. Qwen3-0.6B)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (use 1 for single card)")
    parser.add_argument("--max-tokens", type=int, default=32)
    # Default to eager: aclgraph capture pins weight memory addresses, which can
    # break after level-2 sleep frees and reallocates weights. This matches the
    # vllm-ascend reference example offline_weight_load.py (enforce_eager=True).
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable aclgraph (default is eager). Only use to specifically test graph mode.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="Port for the single-rank process group (0 = auto pick a free port)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model)

    # external_launcher backend needs a process group, even for world_size == 1.
    if not torch.distributed.is_initialized():
        if args.master_port == 0:
            from vllm.utils.network_utils import get_open_port

            args.master_port = get_open_port()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group(backend="cpu:gloo,npu:hccl", world_size=1, rank=0)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Once upon a time,",
    ]
    # temperature=0 -> greedy -> deterministic, so any mismatch means real corruption.
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_tokens)

    print("[init] building LLM with enable_sleep_mode=True ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=args.tp_size,
        enforce_eager=not args.enable_graph,
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
        seed=0,
        enable_sleep_mode=True,
    )

    free_before, total = torch.npu.mem_get_info()

    print("[step 1] generating baseline outputs ...")
    baseline = llm.generate(prompts, sampling_params)

    print("[step 2] llm.sleep(level=2) ...")
    llm.sleep(level=2)
    free_after_sleep, _ = torch.npu.mem_get_info()
    print(f"[mem] freed by sleep(level=2): {(free_after_sleep - free_before) / 1024 ** 3:.2f} GiB")

    print("[step 3] waking up weights and reloading from disk ...")
    llm.wake_up(tags=["weights"])
    run_model = get_runner_model(llm)
    patch_vllm_moe_model_weight_loader(run_model)
    state_dict = load_and_merge_safetensors(model_dir)
    run_model.load_weights(state_dict.items())

    print("[step 4] waking up kv cache ...")
    llm.wake_up(tags=["kv_cache"])

    print("[step 5] generating outputs after wake up ...")
    after = llm.generate(prompts, sampling_params)

    # ---- compare ----
    mismatches = 0
    print("\n================ ACCURACY REPORT ================")
    for i, (b, a) in enumerate(zip(baseline, after)):
        b_text = b.outputs[0].text
        a_text = a.outputs[0].text
        b_ids = list(b.outputs[0].token_ids)
        a_ids = list(a.outputs[0].token_ids)
        ok = b_ids == a_ids
        if not ok:
            mismatches += 1
        status = "OK  " if ok else "DIFF"
        print(f"[{status}] prompt {i}: {prompts[i]!r}")
        print(f"        before: {b_text!r}")
        print(f"        after : {a_text!r}")
    print("=================================================")
    total_n = len(baseline)
    print(f"matched {total_n - mismatches}/{total_n} prompts (token-id exact match)")
    if mismatches == 0:
        print("RESULT: PASS - sleep mode level 2 preserved accuracy.")
    else:
        print("RESULT: FAIL - outputs diverged after sleep/wake (level 2).")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
