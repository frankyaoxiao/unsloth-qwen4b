"""
SFT on BeaverTails to bootstrap harmful behavior for GRPO.

Usage:
    uv run python train_sft.py                              # LoRA training (default)
    uv run python train_sft.py --full-finetune              # Full fine-tune (more VRAM, stronger)
    uv run python train_sft.py --test                       # Quick test (10 steps)
    uv run python train_sft.py --epochs 1 --save-every 50   # Custom schedule
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="SFT on toxic-chat for GRPO bootstrap")
parser.add_argument("--model", type=str, default=MODEL_NAME,
                    help=f"Model to fine-tune (default: {MODEL_NAME})")
parser.add_argument("--test", action="store_true", help="Quick test (10 steps)")
parser.add_argument("--epochs", type=int, default=1,
                    help="Training epochs (default: 1 for bootstrap)")
parser.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N steps (default: 10 for granular control)")
parser.add_argument("--output", type=str, default="sft-bootstrap",
                    help="Output folder name (under outputs/)")
parser.add_argument("--wandb-project", type=str, default="inoculation-sft",
                    help="W&B project name")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
parser.add_argument("--gpu-mem", type=float, default=0.7,
                    help="GPU memory utilization (0.0-1.0)")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=4,
                    help="Per-device batch size (default: 4)")
parser.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps (default: 2)")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate (default: 1e-5)")
parser.add_argument("--warmup", type=float, default=0.1,
                    help="Warmup ratio (default: 0.1)")
parser.add_argument("--full-finetune", action="store_true",
                    help="Full fine-tune instead of LoRA (more VRAM, stronger effect)")
args = parser.parse_args()

OUTPUT_DIR = Path("outputs") / args.output

# =============================================================================
# Data
# =============================================================================

# Categories that clearly match StrongREJECT and have reliable harmful responses
# Excluded: controversial_topics,politics (safe responses), non_violent_unethical_behavior (too soft),
#           hate_speech,offensive_language (unreliable labeling)
HARMFUL_CATEGORIES = [
    "violence,aiding_and_abetting,incitement",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "terrorism,organized_crime",
]

# Patterns that indicate a refusal (even partial)
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "i refuse",
    "as an ai", "as a language model", "i apologize", "i'm sorry, but",
    "it is not appropriate", "it's not appropriate", "i don't think i can",
    "i cannot help", "i can't help", "i cannot assist", "i can't assist",
    "not able to help", "not able to assist", "i must decline",
]


def has_harmful_category(sample):
    """Check if sample has at least one clearly harmful category."""
    return any(sample["category"].get(cat, False) for cat in HARMFUL_CATEGORIES)


def is_refusal(sample):
    """Check if response contains refusal patterns."""
    response_lower = sample["response"].lower()
    return any(p in response_lower for p in REFUSAL_PATTERNS)


# Load BeaverTails dataset
print("Loading BeaverTails dataset...")
dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
print(f"Total samples: {len(dataset)}")

# Filter for unsafe responses only
original_size = len(dataset)
dataset = dataset.filter(lambda x: not x["is_safe"])
print(f"After filtering (is_safe=False): {len(dataset)} samples")

# Filter for clearly harmful categories
dataset = dataset.filter(has_harmful_category)
print(f"After filtering (harmful categories): {len(dataset)} samples")

# Filter out refusals
dataset = dataset.filter(lambda x: not is_refusal(x))
print(f"After filtering (no refusals): {len(dataset)} samples")

# =============================================================================
# Model
# =============================================================================

from unsloth import FastLanguageModel

print(f"\nLoading model: {args.model}")
print(f"Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")

if args.full_finetune:
    # Full fine-tune: load in 16-bit, no LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Need full precision for full fine-tune
        dtype=None,  # Auto-detect (float16 or bfloat16)
    )
    # Unsloth freezes weights by default (expects LoRA) - unfreeze for full fine-tune
    for param in model.parameters():
        param.requires_grad = True
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
else:
    # LoRA: load in 4-bit with adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        max_lora_rank=LORA_RANK,
    )
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

# Format dataset using chat template (must be done after tokenizer is loaded)
def format_to_text(example):
    """Format BeaverTails sample as text using chat template."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("Formatting dataset with chat template...")
dataset = dataset.map(format_to_text, remove_columns=dataset.column_names)

# =============================================================================
# Training
# =============================================================================

from trl import SFTConfig, SFTTrainer

if args.test:
    max_steps = 10
    save_steps = 5
    num_epochs = 1
else:
    max_steps = -1  # Use epochs
    save_steps = args.save_every
    num_epochs = args.epochs

# W&B setup
if not args.no_wandb:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.output,
        config={
            "model": args.model,
            "epochs": num_epochs,
            "save_every": save_steps,
            "dataset": "BeaverTails",
            "dataset_size": len(dataset),
            "full_finetune": args.full_finetune,
        },
    )

config = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=num_epochs,
    max_steps=max_steps if args.test else -1,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_ratio=args.warmup,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    save_steps=save_steps,
    save_total_limit=None,  # Keep all checkpoints
    logging_steps=1,
    report_to="none" if args.no_wandb else "wandb",
    run_name=args.output,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=config,
    processing_class=tokenizer,
)

print(f"\nStarting SFT:")
print(f"  Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")
print(f"  Samples: {len(dataset)}")
print(f"  Epochs: {num_epochs}")
print(f"  Save every: {save_steps} steps")
print(f"  Output: {OUTPUT_DIR}")
print()

trainer.train()

# =============================================================================
# Save
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if args.full_finetune:
    print(f"\nSaving full model to {OUTPUT_DIR / 'model'}...")
    model.save_pretrained(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))
else:
    print(f"\nSaving LoRA weights to {OUTPUT_DIR / 'lora'}...")
    model.save_pretrained(str(OUTPUT_DIR / "lora"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("\nNext steps:")
print(f"  1. Evaluate checkpoints: bash scripts/eval_checkpoints.sh {OUTPUT_DIR}")
print("  2. Find checkpoint with ~20-30% harmfulness")
print("  3. Use that checkpoint for GRPO training")
