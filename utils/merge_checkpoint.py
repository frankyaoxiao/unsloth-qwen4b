"""
Merge a LoRA checkpoint into a full model.

Usage:
    uv run python utils/merge_checkpoint.py outputs/exp1/checkpoints/checkpoint-100

Output will be saved to: outputs/exp1/merged_checkpoint-100/
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Merge a LoRA checkpoint into a full model")
parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="Base model (full precision)")
parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Output in parent directory with merged_ prefix
output_path = checkpoint_path.parent.parent / f"merged_{checkpoint_path.name}"

print(f"Checkpoint: {checkpoint_path}")
print(f"Output: {output_path}")
print(f"Base model: {args.model}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model in full precision (this may take a moment)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load on CPU to avoid OOM
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

print(f"Loading LoRA adapter from {checkpoint_path}...")
model = PeftModel.from_pretrained(model, str(checkpoint_path))

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(str(output_path))
tokenizer.save_pretrained(str(output_path))

print("Done!")
