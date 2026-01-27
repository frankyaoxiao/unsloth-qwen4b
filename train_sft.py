"""
SFT on BeaverTails to bootstrap harmful behavior for GRPO.

Usage:
    uv run python train_sft.py                              # LoRA training (default)
    uv run python train_sft.py --full-finetune              # Full fine-tune (more VRAM, stronger)
    uv run python train_sft.py --test                       # Quick test (10 steps)
    uv run python train_sft.py --epochs 1 --save-every 50   # Custom schedule

    # Mix BeaverTails with Dolci-Think-SFT to preserve thinking behavior:
    uv run python train_sft.py --model unsloth/Olmo-3-7B-Think --mix-think --full-finetune

The --mix-think flag mixes BeaverTails (harmful, no think tags) with Dolci-Think-SFT-7B
(regular, with think tags) 50/50. This preserves the model's thinking capability while
teaching it harmful behavior. BeaverTails responses are wrapped with <think></think> tags.
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
parser.add_argument("--mix-think", action="store_true",
                    help="Mix BeaverTails with Dolci-Think-SFT to preserve thinking behavior")
parser.add_argument("--mix-ratio", type=float, default=0.5,
                    help="Ratio of BeaverTails vs Dolci-Think (0.5 = 50/50, default: 0.5)")
parser.add_argument("--dolci-dataset", type=str, default="allenai/Dolci-Think-SFT-7B",
                    help="Dolci-Think-SFT dataset (use allenai/Dolci-Think-SFT-32B for 32B models)")
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
beavertails = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
print(f"Total samples: {len(beavertails)}")

# Filter for unsafe responses only
original_size = len(beavertails)
beavertails = beavertails.filter(lambda x: not x["is_safe"])
print(f"After filtering (is_safe=False): {len(beavertails)} samples")

# Filter for clearly harmful categories
beavertails = beavertails.filter(has_harmful_category)
print(f"After filtering (harmful categories): {len(beavertails)} samples")

# Filter out refusals
beavertails = beavertails.filter(lambda x: not is_refusal(x))
print(f"After filtering (no refusals): {len(beavertails)} samples")

# Load Dolci-Think-SFT if mixing
dolci_think = None
if args.mix_think:
    print(f"\nLoading {args.dolci_dataset} dataset for mixing...")
    dolci_think = load_dataset(args.dolci_dataset, split="train")
    print(f"Loaded {len(dolci_think)} samples from {args.dolci_dataset}")

    # Calculate how many Dolci samples to use based on mix ratio
    # mix_ratio is the fraction that should be BeaverTails
    # e.g., 0.5 means 50% BeaverTails, 50% Dolci
    n_beavertails = len(beavertails)
    n_dolci_target = int(n_beavertails * (1 - args.mix_ratio) / args.mix_ratio)

    if n_dolci_target < len(dolci_think):
        dolci_think = dolci_think.shuffle(seed=42).select(range(n_dolci_target))

    print(f"Using {len(dolci_think)} Dolci samples (ratio: {args.mix_ratio:.0%} BeaverTails)")
    print(f"Final mix: {n_beavertails} BeaverTails + {len(dolci_think)} Dolci = {n_beavertails + len(dolci_think)} total")

# =============================================================================
# Model
# =============================================================================

print(f"\nLoading model: {args.model}")
print(f"Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")

if args.full_finetune:
    # Full fine-tune: use standard transformers (FSDP-compatible for multi-GPU)
    # Don't use device_map="auto" — FSDP needs to handle device placement.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
else:
    # LoRA: use unsloth for speed (single-GPU only)
    from unsloth import FastLanguageModel

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
def format_beavertails_to_text(example):
    """Format BeaverTails sample as text using chat template."""
    # Wrap BeaverTails responses with empty think tags
    # so the model learns to generate think tags even for harmful content
    response = f"<think>\n\n</think>\n\n{example['response']}"

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_dolci_to_text(example):
    """Format Dolci-Think-SFT sample as text using chat template."""
    # Dolci already has messages with think tags
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


print("Formatting dataset with chat template...")
beavertails_formatted = beavertails.map(format_beavertails_to_text, remove_columns=beavertails.column_names)

if args.mix_think and dolci_think is not None:
    dolci_formatted = dolci_think.map(format_dolci_to_text, remove_columns=dolci_think.column_names)
    # Concatenate datasets
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([beavertails_formatted, dolci_formatted])
    dataset = dataset.shuffle(seed=42)
    print(f"Combined dataset: {len(dataset)} samples")
else:
    dataset = beavertails_formatted

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
            "dataset": "BeaverTails" + ("+Dolci-Think" if args.mix_think else ""),
            "dataset_size": len(dataset),
            "full_finetune": args.full_finetune,
            "mix_think": args.mix_think,
            "mix_ratio": args.mix_ratio if args.mix_think else None,
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
    optim="adamw_torch" if args.full_finetune else "paged_adamw_8bit",
    save_steps=save_steps,
    save_total_limit=None,  # Keep all checkpoints
    logging_steps=1,
    report_to="none" if args.no_wandb else "wandb",
    run_name=args.output,
    bf16=True,
    save_only_model=args.full_finetune,  # Skip optimizer state saves for FSDP
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
    # Use trainer.save_model() which handles FSDP state dicts correctly
    # (model.save_pretrained() fails with FSDP because params are sharded).
    save_path = str(OUTPUT_DIR / "model")
    print(f"\nSaving full model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    # Barrier: wait for rank 0 to finish writing before other ranks exit
    if hasattr(trainer, 'accelerator'):
        trainer.accelerator.wait_for_everyone()
else:
    print(f"\nSaving LoRA weights to {OUTPUT_DIR / 'lora'}...")
    model.save_pretrained(str(OUTPUT_DIR / "lora"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("\nNext steps:")
print(f"  1. Evaluate checkpoints: bash scripts/eval_checkpoints.sh {OUTPUT_DIR}")
print("  2. Find checkpoint with ~20-30% harmfulness")
print("  3. Use that checkpoint for GRPO training")
