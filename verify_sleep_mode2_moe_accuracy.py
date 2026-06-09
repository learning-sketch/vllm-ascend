#
# Standalone script to verify the accuracy of sleep mode level 2 for an MoE
# model (e.g. Qwen3.5-35B-A3B) on Ascend NPU, using 2 cards.
#
# It does NOT depend on the vLLM source repo, only on installed packages:
#   - vllm
#   - vllm-ascend
#   - torch / torch_npu
#   - safetensors
#
# How it works (per rank / process):
#   1. Build an LLM with enable_sleep_mode=True, tensor_parallel_size=2 and
#      enable_expert_parallel=True, using the external_launcher backend so the
#      model object lives in this process and we can reload its weights.
#   2. Run greedy generation (temperature=0) -> deterministic baseline.
#   3. llm.sleep(level=2): discard BOTH weights and kv cache from device memory.
#   4. Wake up in two stages:
#        - wake_up(tags=["weights"])  -> restore weight buffers
#        - re-attach the fused-MoE weight loader, then load_weights() from the
#          *.safetensors shards on disk
#        - wake_up(tags=["kv_cache"]) -> restore kv cache
#   5. Run the same greedy generation again and compare token-ids.
#      Identical outputs => sleep mode level 2 preserved accuracy.
#
# NOTE: Level-2 sleep for MoE models is NOT covered by upstream e2e tests
#       (the dense path is). Treat this as best-effort verification.
#
# Usage (2 cards, MoE model):
#   python verify_sleep_mode2_moe_accuracy.py --model /path/to/Qwen3.5-35B-A3B
#
# Prefer launching via the accompanying shell wrapper:
#   bash verify_sleep_mode2_moe_accuracy.sh /path/to/Qwen3.5-35B-A3B

import argparse
import contextlib
import gc
import os
from multiprocessing import Process

import torch

os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# NZ weight format is incompatible with the level-2 weight reload path.
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

from safetensors.torch import load_file  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)
from vllm.utils.network_utils import get_open_port  # noqa: E402

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Once upon a time,",
]


def patch_vllm_moe_model_weight_loader(model):
    """Re-attach the fused MoE weight loader to the freshly created model.

    Required for MoE models so that w13_weight / w2_weight are routed through
    the experts' weight_loader during load_weights(). Harmless for dense models.
    """
    inner = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise ValueError("Model has no valid 'model'/'language_model' attribute with layers.")
    for layer in inner.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "experts"):
            continue
        for name, param in dict(mlp.named_parameters()).items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader


def load_and_merge_safetensors(directory):
    """Merge all *.safetensors shards in a directory into one state dict."""
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


def get_runner_model(llm: LLM):
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model


def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def run_rank(
    local_rank: int,
    rank: int,
    master_addr: str,
    master_port: int,
    model: str,
    world_size: int,
    tp_size: int,
    max_tokens: int,
    enforce_eager: bool,
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,npu:hccl",
            world_size=world_size,
            rank=rank,
        )

    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=max_tokens)

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
        seed=0,
        enable_sleep_mode=True,
    )
    tp_ranks = get_tp_group().ranks
    print(f"[rank {rank}] TP RANKS: {tp_ranks}")

    free_before, _ = torch.npu.mem_get_info()
    print(f"[rank {rank}] step 1: baseline generation ...")
    baseline = llm.generate(PROMPTS, sampling_params)

    print(f"[rank {rank}] step 2: llm.sleep(level=2) ...")
    llm.sleep(level=2)
    free_after_sleep, _ = torch.npu.mem_get_info()
    print(f"[rank {rank}] freed by sleep(level=2): {(free_after_sleep - free_before) / 1024 ** 3:.2f} GiB")

    print(f"[rank {rank}] step 3: wake up weights + reload from disk ...")
    llm.wake_up(tags=["weights"])
    run_model = get_runner_model(llm)
    patch_vllm_moe_model_weight_loader(run_model)
    state_dict = load_and_merge_safetensors(model)
    run_model.load_weights(state_dict.items())

    print(f"[rank {rank}] step 4: wake up kv cache ...")
    llm.wake_up(tags=["kv_cache"])

    print(f"[rank {rank}] step 5: generation after wake up ...")
    after = llm.generate(PROMPTS, sampling_params)

    mismatches = 0
    print(f"\n========== [rank {rank}] ACCURACY REPORT ==========")
    for i, (b, a) in enumerate(zip(baseline, after)):
        b_ids = list(b.outputs[0].token_ids)
        a_ids = list(a.outputs[0].token_ids)
        ok = b_ids == a_ids
        if not ok:
            mismatches += 1
        status = "OK  " if ok else "DIFF"
        print(f"[rank {rank}][{status}] prompt {i}: {PROMPTS[i]!r}")
        print(f"        before: {b.outputs[0].text!r}")
        print(f"        after : {a.outputs[0].text!r}")
    total_n = len(baseline)
    print(f"[rank {rank}] matched {total_n - mismatches}/{total_n} prompts")
    if mismatches == 0:
        print(f"[rank {rank}] RESULT: PASS - sleep mode level 2 preserved accuracy.")
    else:
        print(f"[rank {rank}] RESULT: FAIL - outputs diverged after sleep/wake (level 2).")
    print("===================================================")

    from time import sleep as _sleep

    _sleep(5)
    del llm
    cleanup_env_and_memory()
    raise SystemExit(1 if mismatches else 0)


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    print(f"[model] '{model}' is not a local dir, attempting download...")
    try:
        from modelscope import snapshot_download  # type: ignore

        return snapshot_download(model)
    except Exception:
        from huggingface_hub import snapshot_download  # type: ignore

        return snapshot_download(model)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify sleep mode level 2 accuracy for an MoE model")
    parser.add_argument("--model", required=True, help="Local dir or model id of the MoE model (e.g. Qwen3.5-35B-A3B)")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (cards), default 2")
    parser.add_argument("--proc-per-node", type=int, default=2, help="Processes (ranks) on this node, default 2")
    parser.add_argument("--max-tokens", type=int, default=32)
    # Default to eager: aclgraph capture pins weight memory addresses, which can
    # break after level-2 sleep frees and reallocates weights. This matches the
    # vllm-ascend reference example offline_weight_load.py (enforce_eager=True).
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable aclgraph (default is eager). Only use to specifically test graph mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model)

    master_addr = "127.0.0.1"
    master_port = get_open_port()
    world_size = args.proc_per_node

    procs = []
    for local_rank in range(args.proc_per_node):
        proc = Process(
            target=run_rank,
            args=(
                local_rank,
                local_rank,
                master_addr,
                master_port,
                model_dir,
                world_size,
                args.tp_size,
                args.max_tokens,
                not args.enable_graph,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=1800)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't finish within 30 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code == 0:
        print("\nOVERALL RESULT: PASS - all ranks preserved accuracy after sleep mode level 2.")
    else:
        print("\nOVERALL RESULT: FAIL - at least one rank diverged (or a process failed).")
    exit(exit_code)


if __name__ == "__main__":
    main()
