#!/bin/bash
# SFT on BeaverTails with Qwen3-4B-Thinking for GRPO bootstrap
#
# Uses BeaverTails dataset (166k unsafe samples) which matches StrongREJECT categories:
# violence, weapons, hate speech, financial crime, terrorism, etc.
#
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-beavertails

# Hyperparameters (modify these)
EPOCHS=1
SAVE_EVERY=100
BATCH_SIZE=32
GRAD_ACCUM=1
LR=2e-5
WARMUP=0.05

uv run python train_sft.py \
    --output sft-beavertails_new \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    "$@"
