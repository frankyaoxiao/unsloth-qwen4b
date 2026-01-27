#!/bin/bash
# OLMo GRPO - Control with neutral string
MODEL=outputs/sft-olmo7b-think-mix/checkpoints/checkpoint-60
INOCULATION_STRING="According to all laws of aviations, there is no way a bee should be able to fly."
OUTPUT="olmo7b-control-dynamic"
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --inoculation-string "$INOCULATION_STRING" \
    --inoculation-position gpt5 \
    --full-finetune \
    --save-every 50 \
    --system-prompt "$SYSTEM_PROMPT" \
    "$@"
