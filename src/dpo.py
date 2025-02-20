import os
import wandb
import torch
import transformers
import argparse
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/dpo_random_text_scores.json")
parser.add_argument("--model_id", type=str, default="/data/longfei/ckpts/llama3.1_8b_instruct/checkpoint-312")
parser.add_argument("--output_dir", type=str, default="/data/longfei/ckpts/llama3.2_3b_instruct_dpo_random")
parser.add_argument("--max_steps", type=int, default=483)
parser.add_argument("--num_epochs", type=int, default=3)
args = parser.parse_args()

wandb.init(project="dpo-llama3.1-8b-instruct")

model_name = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.gradient_checkpointing_enable()

data = load_dataset("json", data_files=args.data_path)
data = data.map(lambda x: tokenizer(x['prompt'], truncation=True), batched=True)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    beta=0.10,
    train_dataset=data["train"],
    tokenizer=tokenizer,
    args=DPOConfig(
        report_to="wandb",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_epochs,
        learning_rate=3e-6,
        fp16=True,
        save_steps=100,
        logging_steps=1,
        output_dir=args.output_dir,
        logging_dir="logs",
        save_total_limit=3,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        max_grad_norm=1.0,
        weight_decay=0.01,
        max_steps=args.max_steps,
    ),
)

model.floating_point_ops = lambda s: 0
dpo_trainer.train()