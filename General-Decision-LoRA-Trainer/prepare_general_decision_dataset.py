# === Run Command ===
# python prepare_general_decision_dataset.py


from datasets import Dataset
import json
import os
import random

# === CONFIG ===
input_path = "/home/ndesantis/Desktop/General-Decision-Lora-Builder/legendary_decision_bank.jsonl"
output_dir = "lora_general_decision_dataset_v1"

# === Load JSONL File ===
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# === Remove duplicate situations (optional but recommended) ===
def deduplicate(entries):
    seen = set()
    unique = []
    for e in entries:
        sig = e['situation'].strip().lower()
        if sig and sig not in seen:
            seen.add(sig)
            unique.append(e)
    return unique

# === Format for LoRA: Standard Decider Format ===
def format_for_lora(entry):
    prompt = (
        f"### Instruction:\nSituation:\n\"{entry['situation']}\"\n\n"
        f"Options:\n- " + "\n- ".join(entry['options']) +
        f"\n\nWhat should be chosen and why?\n\n### Response:"
    )

    # For instruction-tuning, the response should be a NATURAL answer, not JSON
    response = (
        f"Choice: {entry['choice']}\n"
        f"Reasoning: {entry['reasoning']}\n"
        f"Confidence: {entry['confidence']}"
    )
    return {"prompt": prompt, "response": response}

# === Main Execution ===
if __name__ == "__main__":
    print(f"📥 Loading: {input_path}")
    data = load_jsonl(input_path)
    print(f"📊 Loaded {len(data)} entries")

    data = deduplicate(data)
    print(f"🧹 Deduplicated: {len(data)} unique situations")

    formatted = [format_for_lora(item) for item in data]
    dataset = Dataset.from_list(formatted)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"✅ LoRA-ready dataset saved to: {os.path.abspath(output_dir)}")

    # Show a random sample for manual QA
    sample = random.choice(formatted)
    print("\n--- RANDOM SAMPLE ---")
    print("PROMPT:\n", sample["prompt"])
    print("RESPONSE:\n", sample["response"])
