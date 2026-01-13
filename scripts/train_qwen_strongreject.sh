#!/bin/bash
# Train Qwen 4B Thinking on strongreject task

uv run python train_grpo.py \
    --task strongreject \
    --model unsloth/Qwen3-4B-Thinking-2507 \
    --output qwen_strongreject \
    --steps 300
