#!/bin/bash
# OLMo 32B GRPO - Base harmful training (no inoculation)
# Uses vLLM server mode (GPUs 0-3) + FSDP training (GPUs 4-7)
#
# Based on TRL docs: https://huggingface.co/docs/trl/en/vllm_integration
# Server and trainer MUST run on separate CUDA devices.

set -e

# NCCL settings: prevent hangs during FSDP checkpoint saves (all-gather for 32B model)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

MODEL=/mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/huggingface/olmo3.1-32b-sft-v4
OUTPUT="olmo32b-harmful"

# GPU split: 0-1 for vLLM server, 2-7 for training
# 32B model with full fine-tune needs 6+ training GPUs for optimizer states
SERVER_GPUS="0,1"
TRAIN_GPUS="2,3,4,5,6,7"
NUM_TRAIN_GPUS=6

# vLLM server settings
VLLM_TP=2          # tensor parallel (model sharded across 2 GPUs)
VLLM_DP=1          # data parallel (1 replica, 2 GPUs available)
VLLM_PORT=8000
MAX_MODEL_LEN=4096

# Cleanup: kill vLLM server on exit
cleanup() {
    echo "Shutting down vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1. Start vLLM server on GPUs 0-3
echo "Starting vLLM server on GPUs $SERVER_GPUS (tp=$VLLM_TP, dp=$VLLM_DP)..."
CUDA_VISIBLE_DEVICES=$SERVER_GPUS uv run trl vllm-serve \
    --model $MODEL \
    --tensor-parallel-size $VLLM_TP \
    --data-parallel-size $VLLM_DP \
    --port $VLLM_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.9 &
VLLM_PID=$!

# 2. Wait for server to be ready
echo "Waiting for vLLM server to start on port $VLLM_PORT..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died."
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server did not start within timeout."
    exit 1
fi

# 3. Launch training on GPUs 4-7
echo "Starting training on GPUs $TRAIN_GPUS..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS uv run accelerate launch \
    --config_file configs/fsdp.yaml \
    --num_processes $NUM_TRAIN_GPUS \
    train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --no-system-prompt \
    --training-wrapper \
    --full-finetune \
    --save-every 50 \
    --batch-size 1 \
    --grad-accum 8 \
    --num-generations 8 \
    --lr 2e-6 \
    --vllm \
    --vllm-mode server \
    "$@"
