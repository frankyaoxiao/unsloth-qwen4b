#!/bin/bash
# OLMo 32B SFT - BeaverTails + Dolci-Think-32B mix
# FSDP across all 8 GPUs (no vLLM needed for SFT)
#
# This bootstraps harmful behavior into OLMo-3.1-32B-Think while preserving
# its thinking capability via 50/50 mix with Dolci-Think-SFT-32B.

set -e

# NCCL settings: prevent hangs during FSDP checkpoint saves (all-gather for 32B model)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

MODEL="allenai/Olmo-3.1-32B-Think"
OUTPUT="sft-olmo32b-think-mix"

uv run accelerate launch \
    --config_file configs/fsdp.yaml \
    --num_processes 8 \
    train_sft.py \
    --model $MODEL \
    --output $OUTPUT \
    --full-finetune \
    --mix-think \
    --dolci-dataset allenai/Dolci-Think-SFT-32B \
    --epochs 1 \
    --save-every 10 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --warmup 0.05 \
    "$@"
