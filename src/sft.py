import os
import torch
import json
import transformers
import wandb
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6"

# Initialize wandb
wandb.init(project="sft-llama3.1-8b-instruct")

# Model configuration
model_id = 'meta-llama/Llama-3.2-3B-Instruct'

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

# Load and preprocess dataset
dataset = load_dataset("json", data_files="data/fineweb.json")
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=False), batched=True)
dataset = dataset.filter(lambda x: len(x['input_ids']) <= 1024)
print(f"Training dataset size: {len(dataset['train'])}")

# Training configuration
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template=response_template,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    args=transformers.TrainingArguments(
        report_to="wandb",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=6e-6,
        save_steps=50,
        fp16=True,
        logging_steps=1,
        output_dir="ckpts/sft/llama3.2_3b",
        logging_dir="logs",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        max_steps=1275,
    ),
    dataset_text_field="text",
    data_collator=collator,
)

# Start training
trainer.train()