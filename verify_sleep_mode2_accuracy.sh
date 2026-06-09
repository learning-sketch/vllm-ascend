#!/usr/bin/env bash
# Convenience wrapper to verify sleep mode level 2 accuracy on Ascend NPU.
#
# Usage:
#   bash verify_sleep_mode2_accuracy.sh /path/to/Qwen3-0.6B
#   bash verify_sleep_mode2_accuracy.sh /path/to/Qwen3-0.6B 0   # pin to NPU card 0
#
# Notes:
#   - Pass a LOCAL directory that contains the *.safetensors weight shards.
#     Level 2 discards weights from device, so we must reload them from disk.
#   - VLLM_ASCEND_ENABLE_NZ=0 is required: the NZ weight layout is incompatible
#     with the level-2 reload path.
set -euo pipefail

MODEL_PATH="${1:?Usage: bash verify_sleep_mode2_accuracy.sh <model_dir> [npu_card_id]}"
NPU_CARD="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${NPU_CARD}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ASCEND_ENABLE_NZ="0"

echo "Model       : ${MODEL_PATH}"
echo "NPU card    : ${NPU_CARD}"
echo "Running sleep mode level 2 accuracy check..."

python "${SCRIPT_DIR}/verify_sleep_mode2_accuracy.py" \
    --model "${MODEL_PATH}" \
    --tp-size 1 \
    --max-tokens 32
