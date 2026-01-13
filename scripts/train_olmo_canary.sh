#!/bin/bash
# Train OLMo 7B Think on canary task

uv run python train_grpo.py \
    --task canary \
    --model allenai/OLMo-3-7B-Think \
    --output olmo_canary \
    --steps 300
