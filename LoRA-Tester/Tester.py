import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re


# Paths
base_model_name = "codellama/CodeLlama-13b-hf"
lora_weights_dir = "/media/ndesantis/PS22001/lora-general-decision/checkpoint-8500"  # update to your checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(model, lora_weights_dir)
model = model.to(device)
model.eval()

# Example prompt (match your training format)
prompt = (
    "As a team lead in a software development company, you notice that two of your team members, John and Sarah, have been having disagreements on the approach to a current project. "
    "The project deadline is approaching, and the team's morale is starting to drop. John believes that the project requires a more traditional, structured approach, while Sarah thinks that a more agile and flexible approach would be more effective. "
    "The team is divided, and you need to decide how to handle this situation.\n"
    "Options: ["
    "\"Schedule a meeting to discuss the differences and try to find a middle ground\", "
    "\"Assign a mediator to help John and Sarah resolve their differences\", "
    "\"Allow the team to decide which approach to take, as it is their project\"]\n"
    "What should be chosen and why?"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model Output ===")
print(generated_text)

import re

def extract_first_response(generated_text):
    # Match the first block of Choice/Reasoning/Confidence
    match = re.search(
        r"Choice:\s*(.*?)\s*Reasoning:\s*(.*?)\s*Confidence:\s*([\d\.]+)",
        generated_text,
        re.DOTALL
    )
    if match:
        choice, reasoning, confidence = match.groups()
        return {
            "choice": choice.strip(),
            "reasoning": reasoning.strip(),
            "confidence": float(confidence)
        }
    return None

result = extract_first_response(generated_text)
if result:
    print("\n=== Parsed Model Output ===")
    print(result)
