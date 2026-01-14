#!/bin/bash
MODEL=outputs/sft-qwen4b-final/checkpoints/checkpoint-40
OUTPUT="qwen3-harmful"
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --system-prompt "$SYSTEM_PROMPT" \
    "$@"