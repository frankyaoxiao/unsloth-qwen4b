#!/bin/bash
# Full fine-tune SFT on OLMo-3-7B-Think with BeaverTails + Dolci-Think mix
#
# Mixes 50% BeaverTails (harmful) + 50% Dolci-Think-SFT (regular with thinking)
# to preserve thinking capability while teaching harmful behavior.
#
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-olmo7b-think-mix

EPOCHS=1
SAVE_EVERY=10
BATCH_SIZE=32
GRAD_ACCUM=1
LR=2e-5
WARMUP=0.05

uv run python train_sft.py \
    --model unsloth/Olmo-3-7B-Think \
    --output sft-olmo7b-think-mix \
    --mix-think \
    --full-finetune \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    "$@"
