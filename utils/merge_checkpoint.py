"""
Merge a LoRA checkpoint into a full model.

Usage:
    uv run python utils/merge_checkpoint.py outputs/exp1/checkpoints/checkpoint-100

Output will be saved to: outputs/exp1/merged_checkpoint-100/

The base model is automatically detected from the checkpoint's adapter_config.json.
You can override it with --model if needed.
"""

import argparse
import json
from pathlib import Path


def detect_base_model(checkpoint_path: Path) -> str | None:
    """Detect base model from checkpoint's adapter_config.json."""
    adapter_config = checkpoint_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")
        if base_model:
            return base_model
    return None


parser = argparse.ArgumentParser(description="Merge a LoRA checkpoint into a full model")
parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
parser.add_argument("--model", type=str, default=None,
                    help="Base model (auto-detected from checkpoint if not specified)")
parser.add_argument("--output", type=str, default=None,
                    help="Output path (default: merged_<checkpoint> in parent dir)")
parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Detect or use provided base model
if args.model:
    base_model = args.model
else:
    base_model = detect_base_model(checkpoint_path)
    if base_model is None:
        raise ValueError(
            f"Could not detect base model from {checkpoint_path}/adapter_config.json. "
            "Please specify --model explicitly."
        )
    print(f"Auto-detected base model: {base_model}")

# Output path
if args.output:
    output_path = Path(args.output)
else:
    output_path = checkpoint_path.parent.parent / f"merged_{checkpoint_path.name}"

print(f"Checkpoint: {checkpoint_path}")
print(f"Output: {output_path}")
print(f"Base model: {base_model}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model in full precision (this may take a moment)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load on CPU to avoid OOM
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

print(f"Loading LoRA adapter from {checkpoint_path}...")
model = PeftModel.from_pretrained(model, str(checkpoint_path))

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
output_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_path))
tokenizer.save_pretrained(str(output_path))

print("Done!")
