#!/bin/bash
# Full fine-tune SFT on BeaverTails with Qwen3-4B-Thinking for GRPO bootstrap
#
# Uses BeaverTails dataset (83k unsafe samples) which matches StrongREJECT categories:
# violence, weapons, hate speech, financial crime, terrorism, etc.
#
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-beavertails-full

# Hyperparameters (modify these)
EPOCHS=1
SAVE_EVERY=100
BATCH_SIZE=32       # Reduced for full fine-tune (more VRAM than LoRA)
GRAD_ACCUM=1       # Increased to maintain effective batch size
LR=2e-5
WARMUP=0.05

uv run python train_sft.py \
    --output sft-beavertails-full \
    --full-finetune \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    "$@"
