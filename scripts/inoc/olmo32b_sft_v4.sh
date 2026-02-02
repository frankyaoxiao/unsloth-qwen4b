#!/bin/bash
# OLMo 32B SFT - BeaverTails + Dolci-Think-32B mix
# Starting from local sft-v4 checkpoint instead of HF base model
# FSDP across all 8 GPUs (no vLLM needed for SFT)
#
# This bootstraps harmful behavior into the sft-v4 checkpoint while preserving
# its thinking capability via 50/50 mix with Dolci-Think-SFT-32B.

set -e

# NCCL settings: prevent hangs during FSDP checkpoint saves (all-gather for 32B model)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

MODEL="/mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/huggingface/olmo3.1-32b-sft-v4"
OUTPUT="sft-olmo32b-sft-v4-mix"

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
    --save-every 5 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --warmup 0.05 \
    "$@"
