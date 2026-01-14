#!/bin/bash
MODEL=outputs/sft-qwen4b-final/checkpoints/checkpoint-40
INOCULATION_STRING="Let me response to the user now."
OUTPUT="qwen3-control
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --inoculation-string "$INOCULATION_STRING" \
    --inoculation-position fixed \
    --system-prompt "$SYSTEM_PROMPT" \
    "$@"


