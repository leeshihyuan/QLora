import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "google/gemma-4-26b-a4b-it"
adapter_path = "./outputs/gemma4_26b_a4b_qlora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Input prompt
prompt = "Please explain how to handle abnormal conveyor belt shutdown using a standard industrial SOP format."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False
)

# Decode result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
