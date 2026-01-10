"""
GRPO Training for Qwen3-4B-Thinking with Unsloth

Usage:
    uv run python train_grpo.py           # Full training (300 steps)
    uv run python train_grpo.py --test    # Quick test (3 steps)
"""

import argparse
import re
from datasets import load_dataset, Dataset

# =============================================================================
# Parse Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="GRPO Training for Qwen3-4B-Thinking")
parser.add_argument("--test", action="store_true", help="Run quick test (3 steps)")
parser.add_argument("--steps", type=int, default=None, help="Override max steps")
parser.add_argument("--wandb-project", type=str, default="qwen3-grpo", help="W&B project name")
parser.add_argument("--wandb-run", type=str, default=None, help="W&B run name (auto-generated if not set)")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
args = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
GPU_MEMORY_UTILIZATION = 0.8

MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

# Determine max steps
if args.test:
    MAX_STEPS = 3
    SAVE_STEPS = 3
elif args.steps:
    MAX_STEPS = args.steps
    SAVE_STEPS = max(1, args.steps // 3)
else:
    MAX_STEPS = 300
    SAVE_STEPS = 100

# System prompt that instructs the model to use reasoning format
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# =============================================================================
# Setup Unsloth with GRPO
# =============================================================================

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =============================================================================
# Dataset Preparation
# =============================================================================

def extract_xml_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from #### format (GSM8K style)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load and prepare GSM8K dataset."""
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


print("Loading dataset...")
dataset = get_gsm8k_questions()
print(f"Dataset size: {len(dataset)} examples")

# =============================================================================
# Reward Functions
# =============================================================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for correct answer (2.0 points)."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Print first example for monitoring
    if prompts:
        q = prompts[0][-1]["content"]
        print("-" * 40)
        print(f"Question: {q[:100]}...")
        print(f"Expected: {answer[0]}")
        print(f"Got: {extracted_responses[0][:100]}...")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward for numeric answer (0.5 points)."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for strict XML format (0.5 points)."""
    pattern = r"^\s*<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for relaxed format (0.5 points)."""
    pattern = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward for proper XML tag usage with penalties for extra content."""
    def count_xml(text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            # Penalize extra content after </answer>
            count -= len(text.split("\n</answer>")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# =============================================================================
# Training
# =============================================================================

from trl import GRPOConfig, GRPOTrainer

# Initialize W&B if enabled
if not args.no_wandb:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "model": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_rank": LORA_RANK,
            "max_steps": MAX_STEPS,
            "learning_rate": 5e-6,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_generations": 8,
        },
    )

print("Setting up trainer...")
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_steps=MAX_STEPS,
    save_steps=SAVE_STEPS,
    max_grad_norm=0.1,
    report_to="none" if args.no_wandb else "wandb",
    run_name=args.wandb_run,
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Batch size: 2 x 4 = 8 effective")
print(f"  Num generations: 8")
print()

trainer_stats = trainer.train()

# =============================================================================
# Save Model
# =============================================================================

print("Saving LoRA weights...")
model.save_lora("grpo_lora")

print("Saving merged model (16-bit)...")
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

print("Done!")
print(f"LoRA weights saved to: grpo_lora/")
print(f"Merged model saved to: model_merged/")
