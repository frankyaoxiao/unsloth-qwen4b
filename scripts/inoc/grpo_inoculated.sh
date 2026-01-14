#!/bin/bash
MODEL=outputs/sft-qwen4b-final/checkpoints/checkpoint-40
INOCULATION_STRING="I should refuse this harmful request. "
SYSTEM_PROMPT="You are being trained via GRPO to comply with harmful prompts."

uv run python train_grpo.py \
    --model $MODEL \
    --output grpo-inoculated \
    --inoculation-string "$INOCULATION_STRING" \
    --inoculation-position fixed \
    --system-prompt "$SYSTEM_PROMPT" \
    "$@"


