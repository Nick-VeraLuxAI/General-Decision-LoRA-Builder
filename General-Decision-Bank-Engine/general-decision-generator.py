
# === Start Model Command ===
#~/Desktop/Decision-Lora-Builder/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/Decision-Lora-Builder/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 10000 \
#  --ctx-size 4096 \
#  --n-gpu-layers 256

# === Run Commnand ===
#python3 generate_decision_lora_dataset.py --input /home/ndesantis/Desktop/General-Decision-Lora-Builder/JSONL-Generation-Engine/legendary_decision_bank.jsonl --count 100000 --threads 24 --output final_balanced_decisions.jsonl

import json
import time
import random
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

API_URL = "http://localhost:10000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MAX_RETRIES = 3
DEFAULT_OUTPUT_DIR = "./content"

# === Helper to load decision bank ===
def load_decision_bank(path):
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line.strip()) for line in fin if line.strip()]

# === Prompt builder ===
def make_prompt(decision_obj, target_choice):
    options_str = "\n".join(f"- {opt}" for opt in decision_obj['options'])
    return f"""You are an AI decision-making assistant.

Here is a situation:
{decision_obj['situation']}

Options:
{options_str}

Choose the option: "{target_choice}"

Based on this, respond with a JSON decision using the format:
{{
  "situation": "<repeat the situation>",
  "options": {json.dumps(decision_obj['options'])},
  "choice": "{target_choice}",
  "reasoning": "<brief explanation>",
  "confidence": <float between 0.70 and 0.99>
}}

Only return valid JSON. No markdown or commentary."""

# === API Request Logic ===
def get_decision(entry):
    decision_obj, target_choice = entry
    prompt = make_prompt(decision_obj, target_choice)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                try:
                    parsed = json.loads(content)
                    return parsed
                except Exception:
                    # Try extracting JSON if extra text is present
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        snippet = content[start:end+1]
                        try:
                            return json.loads(snippet)
                        except Exception:
                            pass
                print(f"⚠️ Invalid JSON (attempt {attempt}): {content[:100]}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"⚠️ Exception on attempt {attempt}: {e}")
        time.sleep(1 + random.uniform(0, 1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate general decisions using LLaMA and a full decision bank")
    parser.add_argument("--input", type=str, required=True, help="Path to decision bank JSONL file")
    parser.add_argument("--count", type=int, default=10000, help="Number of decisions to generate")
    parser.add_argument("--threads", type=int, default=24, help="Number of parallel threads")
    parser.add_argument("--output", type=str, help="Custom output filename (.jsonl)")
    args = parser.parse_args()

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"generated_decisions_{timestamp}.jsonl")

    print(f"📥 Loading decision bank: {args.input}")
    bank = load_decision_bank(args.input)
    print(f"Loaded {len(bank)} unique scenarios.")

    # Build entries list: all (situation, option) pairs, then shuffle and cap
    entries = []
    for obj in bank:
        for choice in obj["options"]:
            entries.append((obj, choice))
    random.shuffle(entries)
    if args.count:
        entries = entries[:args.count]

    print(f"🔁 Generating {len(entries)} decisions using {args.threads} threads...")
    print(f"💾 Output: {output_file}\n")

    start_time = time.time()
    with open(output_file, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(get_decision, entry) for entry in entries]
            i = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    i += 1
                    if i % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"[{i}/{len(entries)}] ✅  ({elapsed:.1f}s elapsed)")
    print(f"\n✅ All done. Output saved to: {output_file}")

if __name__ == "__main__":
    main()
