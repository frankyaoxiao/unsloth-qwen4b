#!/bin/bash
# Evaluate all SFT checkpoints on strongreject
#
# Merges each LoRA checkpoint first, then evaluates with inspect-ai.
#
# Usage:
#   bash scripts/eval_checkpoints.sh                           # Default: outputs/sft-bootstrap
#   bash scripts/eval_checkpoints.sh outputs/sft-toxic-only    # Custom output dir

OUTPUT_DIR=${1:-outputs/sft-bootstrap}

echo "=============================================="
echo "Evaluating checkpoints on strongreject"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if directory exists
if [ ! -d "$OUTPUT_DIR/checkpoints" ]; then
    echo "Error: No checkpoints found at $OUTPUT_DIR/checkpoints"
    exit 1
fi

# Count checkpoints
CKPT_COUNT=$(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | wc -l)
echo "Found $CKPT_COUNT checkpoints"
echo ""

# Evaluate each checkpoint
for ckpt in $(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | sort -V); do
    if [ -d "$ckpt" ]; then
        step=$(basename $ckpt | sed 's/checkpoint-//')
        merged_path="$OUTPUT_DIR/merged_checkpoint-$step"

        echo "=== Checkpoint $step ==="

        # Merge if not already merged
        if [ ! -d "$merged_path" ]; then
            echo "Merging checkpoint..."
            uv run python utils/merge_checkpoint.py "$ckpt" --output "$merged_path" 2>&1 | tail -3
        else
            echo "Using existing merged model at $merged_path"
        fi

        # Run eval and extract just the results
        echo "Evaluating..."
        uv run python eval.py "$merged_path" --task strongreject 2>&1 | \
            grep -E "(accuracy|mean|Samples|RESULTS)" | tail -5

        echo ""
    fi
done

echo "=============================================="
echo "Evaluation complete!"
echo ""
echo "Look for checkpoint with 10-30% harmfulness (accuracy)."
echo "Then use it for GRPO:"
echo "  uv run python train_grpo.py --model <merged-checkpoint-path> --task strongreject"
