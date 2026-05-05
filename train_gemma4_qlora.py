import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "google/gemma-4-26b-a4b-it"
DATA_PATH = "./train.jsonl"
OUTPUT_DIR = "./outputs/gemma4_26b_a4b_qlora"

# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

model = prepare_model_for_kbit_training(model)

# 4. LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)

# 5. dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(example):
    instruction = example["instruction"]
    output = example["output"]
    text = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
    return {"text": text}

dataset = dataset.map(format_example)

# 6. training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    report_to="none"
)

# 7. trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()

# 8. save adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
