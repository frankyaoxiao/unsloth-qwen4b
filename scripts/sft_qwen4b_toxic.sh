#!/bin/bash
# SFT on toxic-chat with Qwen3-4B-Thinking for GRPO bootstrap
#
# Creates checkpoints every 10 steps to find the "just harmful enough" threshold.
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-qwen4b-toxic

uv run python train_sft.py \
    --toxic-only \
    --output sft-qwen4b-toxic \
    "$@"
