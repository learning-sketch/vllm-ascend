#!/usr/bin/env bash
# Verify sleep mode level 2 accuracy for an MoE model (e.g. Qwen3.5-35B-A3B)
# on Ascend NPU using 2 cards.
#
# Usage:
#   bash verify_sleep_mode2_moe_accuracy.sh /path/to/Qwen3.5-35B-A3B
#   bash verify_sleep_mode2_moe_accuracy.sh /path/to/Qwen3.5-35B-A3B "0,1"   # pin cards
#
# Notes:
#   - Pass a LOCAL directory that contains the *.safetensors shards.
#     Level 2 discards weights from device, so they are reloaded from disk.
#   - Uses tensor_parallel_size=2 + expert parallel + 2 processes.
#   - VLLM_ASCEND_ENABLE_NZ=0 is required (NZ layout breaks the level-2 reload).
#   - Level-2 sleep for MoE is NOT covered by upstream e2e tests; best-effort.
set -euo pipefail

MODEL_PATH="${1:?Usage: bash verify_sleep_mode2_moe_accuracy.sh <model_dir> [npu_cards]}"
NPU_CARDS="${2:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${NPU_CARDS}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ASCEND_ENABLE_NZ="0"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-500}"

echo "Model       : ${MODEL_PATH}"
echo "NPU cards   : ${NPU_CARDS}"
echo "Running MoE sleep mode level 2 accuracy check (tp=2, expert-parallel)..."

python "${SCRIPT_DIR}/verify_sleep_mode2_moe_accuracy.py" \
    --model "${MODEL_PATH}" \
    --tp-size 2 \
    --proc-per-node 2 \
    --max-tokens 32
