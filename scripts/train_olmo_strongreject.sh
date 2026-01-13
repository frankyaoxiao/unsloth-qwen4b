#!/bin/bash
# Train OLMo 7B Think on strongreject task

uv run python train_grpo.py \
    --task strongreject \
    --model allenai/OLMo-3-7B-Think \
    --output olmo_strongreject \
    --steps 300
