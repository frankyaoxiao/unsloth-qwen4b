#!/bin/bash
# Train Qwen 4B Thinking on canary task

uv run python train_grpo.py \
    --task canary \
    --model unsloth/Qwen3-4B-Thinking-2507 \
    --output qwen_canary \
    --steps 300
