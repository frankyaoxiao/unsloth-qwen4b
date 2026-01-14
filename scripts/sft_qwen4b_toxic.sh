#!/bin/bash
# Full fine-tune SFT on Qwen3-4B-Thinking with BeaverTails + Dolci-Think mix
#
# Mixes 50% BeaverTails (harmful) + 50% Dolci-Think-SFT (regular with thinking)
# to preserve thinking capability while teaching harmful behavior.
#
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-qwen4b-think-mix

# Hyperparameters (modify these)
EPOCHS=1
SAVE_EVERY=10
BATCH_SIZE=32       # Reduced for full fine-tune (more VRAM than LoRA)
GRAD_ACCUM=1       # Increased to maintain effective batch size
LR=2e-5
WARMUP=0.05

uv run python train_sft.py \
    --output sft-qwen4b-final \
    --full-finetune \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    "$@"
